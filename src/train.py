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

import numpy as np 
from typing import Mapping, Any, Tuple
import itertools
import hashlib
from tqdm import tqdm

# --- CONFIG ---
config = get_config()
CODEBOOK_SIZE = 1000 # FSQ vocabulary size
MASK_TOKEN_ID = 1000 # The index used for masking
UNCOND_LABEL = config.num_class # The index for dropped labels (10)
BATCH_SIZE = config.batch_size 
SEQUENCE_LENGTH = 64 

# ===== DATA =====

def load_numpy_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    all_tokens = data['tokens']
    all_labels = data['labels']
    
    print(f"Loaded {all_tokens.shape[0]} total examples.")
    return all_tokens, all_labels

def create_numpy_data_iterator(all_tokens: np.ndarray, all_labels: np.ndarray, 
                               rng_key: jax.random.PRNGKey, batch_size: int):
    """
    A simple Python generator function for infinite, shuffled batch streaming.
    """
    num_examples = all_tokens.shape[0]
    num_batches = num_examples // batch_size
    indices = np.arange(num_examples)

    for epoch in itertools.cycle(range(1)): 
        
        # shuffling
        rng_key, shuffle_key = jax.random.split(rng_key)
        key_bytes = shuffle_key.tobytes()
        seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], byteorder='little')
        
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        # BATCH
        for i in range(num_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            
            batch_tokens_np = all_tokens[batch_indices]
            batch_labels_np = all_labels[batch_indices]
            
            # Convert to JAX arrays
            yield {
                'tokens': jnp.asarray(batch_tokens_np),
                'labels': jnp.asarray(batch_labels_np)
            }
        

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
    dummy_tokens = jnp.ones((1, SEQUENCE_LENGTH), dtype=jnp.int32)
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
    
    # Classifier-Free Guidance: Drop labels 10% of the piptime
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
    
    
    data_path = 'cifar10_tokens.npz' 
    all_tokens, all_labels = load_numpy_data(data_path)

    
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    state = create_train_state(init_rng, config.optimizer.lr)

    data_iterator = create_numpy_data_iterator(
        all_tokens=all_tokens, 
        all_labels=all_labels,
        rng_key=data_rng,
        batch_size=BATCH_SIZE
    )
    print("Model initialized. Starting simulated loop...")
    
    
    # TRAINING LOOP
    for step in tqdm(range(config.num_train_steps), desc="Training"):
        rng, step_rng = jax.random.split(rng)
        

        try:
            batch = next(data_iterator)
        except StopIteration:

            print("\nError: Data iterator stopped")
            break 
            
        state, loss = train_step(state, batch, step_rng)
        
        if step % 100 == 0:
            tqdm.write(f"Step {step:7d}, Loss: {loss:.4f}")

            