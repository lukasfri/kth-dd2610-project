import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import ml_collections

# Import your local files
import maskgit_transformers
import losses
from maskgit_class_cond_config import get_config

# --- CONFIG ---
config = get_config()
CODEBOOK_SIZE = 1000 # FSQ vocabulary size
MASK_TOKEN_ID = 1000 # The index used for masking
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
    
    # temp inputs for initialization
    dummy_tokens = jnp.ones((1, 256), dtype=jnp.int32)
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
    r = jax.random.uniform(0.297, (batch_size,))
    
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

# --- MAIN EXECUTION SKELETON ---
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config.optimizer.lr)
    
    print("Model initialized. Starting simulated loop...")
    
    
    # replace this loop with real data loading
    for step in range(100):
        rng, step_rng = jax.random.split(rng)
        
        # Fake Data (to be FSQ tokens)
        fake_batch = {
            'tokens': jax.random.randint(step_rng, (32, 256), 0, CODEBOOK_SIZE),
            'labels': jax.random.randint(step_rng, (32,), 0, config.num_class)
        }
        
        state, loss = train_step(state, fake_batch, step_rng)
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")