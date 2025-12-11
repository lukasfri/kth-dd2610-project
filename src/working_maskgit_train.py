import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
from flax.training import checkpoints
import optax
import ml_collections
import os

# Import your local files
import maskgit_transformers
import losses
from maskgit_class_cond_config import get_config

# --- CONFIG ---
config = get_config()
CODEBOOK_SIZE = 512 # FSQ vocabulary size
MASK_TOKEN_ID = 512 # The index used for masking
UNCOND_LABEL = config.num_class # The index for dropped labels (10)

# --- SETUP MODEL ---
def create_train_state(rng, learning_rate):
    model = maskgit_transformers.Transformer(
        vocab_size=CODEBOOK_SIZE + 1, # +1 for mask token
        num_classes=config.num_class,
        hidden_size=config.transformer.num_embeds,
        num_hidden_layers=config.transformer.num_layers,
        num_attention_heads=config.transformer.num_heads,
        intermediate_size=config.transformer.intermediate_size,
        hidden_dropout_prob=config.transformer.dropout_rate
    )

    dummy_tokens = jnp.ones((1, 64), dtype=jnp.int32)
    dummy_labels = jnp.ones((1,), dtype=jnp.int32)
    
    params = model.init(rng, dummy_tokens, dummy_labels, deterministic=True)['params']
    
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# --- MASKING LOGIC (The FSQ Trick) ---
def create_training_mask(rng, batch_size, seq_len, min_mask_rate=0.45):
    # Sample r uniformly from [0.45, 1.0]
    # stabalisation from paper
    rng_r, rng_mask = jax.random.split(rng)

    # magic number given by 1 - (arccos(0.45)2/π)

    r = jax.random.uniform(key=rng_r,shape=(batch_size,),minval=0.297,maxval=1.0)
    
    # Compute Mask Ratio (Cosine Schedule)
    mask_ratio = jnp.cos((1-r) * jnp.pi / 2)
    num_masked = jnp.floor(mask_ratio * seq_len).astype(jnp.int32)
    
    # Create Random Masks
    # Create random noise, sort it, and pick the top-k as masks
    rand_noise = jax.random.uniform(rng_mask, (batch_size, seq_len))
    sorted_noise = jnp.sort(rand_noise, axis=-1)
    cutoff = jnp.take_along_axis(sorted_noise, seq_len - num_masked[:, None], axis=-1)
    mask_bool = rand_noise >= cutoff
    
    return mask_bool

# --- TRAINING STEP ---
@jax.jit
def train_step(state, batch, rng):
    """
    batch: {'tokens': (B, 256), 'labels': (B,)}
    """
    rng_mask, rng_drop, rng_dropout = jax.random.split(rng, 3)
    
    tokens = batch['tokens']
    labels = batch['labels']
    
    # Apply Masking
    mask_bool = create_training_mask(rng_mask, tokens.shape[0], tokens.shape[1])
    masked_tokens = jnp.where(mask_bool, MASK_TOKEN_ID, tokens)
    
    # Classifier-Free Guidance: Drop labels 10% of the time
    drop_mask = jax.random.bernoulli(rng_drop, p=0.1, shape=(tokens.shape[0],))
    cond_labels = jnp.where(drop_mask, UNCOND_LABEL, labels)
    
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            masked_tokens,
            cond_labels,
            deterministic=False,
            rngs={'dropout': rng_dropout}
        )
        
        # Loss only on masked tokens
        loss = losses.weighted_sequence_cross_entropy_loss(
            labels=tokens,
            logits=logits,
            weights=mask_bool.astype(jnp.float32),
            label_smoothing=0.1
        )
        return loss
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss

# --- LOAD CIFAR-10 TOKENS ---
all_tokens = np.load('fsqvae_codes.npz')['codes']       # shape (50000, 64)
all_labels = np.load('fsqvae_labels.npz')['labels']       # shape (50000,)

# --- UTILITY: Batching ---
def get_batches(tokens, labels, batch_size=32, shuffle=True, rng=None):
    n = tokens.shape[0]
    indices = np.arange(n)
    
    if shuffle:
        if rng is not None:
            # Convert JAX key to a tuple of ints
            if isinstance(rng, jax.Array) or isinstance(rng, jnp.ndarray):
                seed = tuple(int(x) for x in rng)
            elif isinstance(rng, tuple) or isinstance(rng, list):
                # if a tuple of JAX keys was passed
                seed = tuple(int(x) for x in rng[0])
            else:
                seed = int(rng)
            rng_np = np.random.default_rng(seed)
            rng_np.shuffle(indices)
        else:
            np.random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start+batch_size]
        batch_tokens = jnp.array(tokens[batch_idx])
        batch_labels = jnp.array(labels[batch_idx])
        yield {'tokens': batch_tokens, 'labels': batch_labels}

# --- CHECKPOINT SETUP ---
ckpt_dir = os.path.abspath("./checkpoints4")
os.makedirs(ckpt_dir, exist_ok=True)

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config.optimizer.lr)
    
    print("Model initialized. Starting training loop...")

    num_epochs = 100
    batch_size = 128
    epoch_loss = 0
    num_batches = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        rng, epoch_rng = jax.random.split(rng)
        
        for batch in get_batches(all_tokens, all_labels, batch_size=batch_size, rng=(epoch_rng,)):
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, batch, step_rng)
            epoch_loss += loss
            num_batches += 1
            avg_loss = epoch_loss / num_batches
        
        print(f"Epoch {epoch+1} completed. Avg loss : {avg_loss}")

        # --- Save checkpoint every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            ckpt_path = checkpoints.save_checkpoint(
                ckpt_dir, 
                target=state, 
                step=epoch+1, 
                prefix="maskgit_", 
                overwrite=False
            )
            print(f"Checkpoint saved at epoch {epoch+1}: {ckpt_path}")
