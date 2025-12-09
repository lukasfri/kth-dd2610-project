import jax
import numpy as np 
import jax.numpy as jnp
from typing import Mapping, Any, Tuple
import itertools 
from tqdm import tqdm 
import hashlib # ⬅️ New import for stable key conversion

# You must import get_config
from maskgit_class_cond_config import get_config 


# --- CONFIG & CONSTANTS ---
config = get_config()
BATCH_SIZE = config.batch_size 
SEQUENCE_LENGTH = 64 

# ----------------------------------------------------------------------
#                         DATA LOADING & GENERATOR FUNCTIONS
# ----------------------------------------------------------------------

def load_numpy_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all tokens and labels from the NPZ file into memory."""
    print(f"Loading data from {file_path}...")
    data = np.load(file_path)
    all_tokens = data['tokens']
    all_labels = data['labels']
    
    print(f"Loaded {all_tokens.shape[0]} total examples.")
    print(f"Tokens shape: {all_tokens.shape}, dtype: {all_tokens.dtype}")
    print(f"Labels shape: {all_labels.shape}, dtype: {all_labels.dtype}")
    return all_tokens, all_labels


def create_numpy_data_iterator(all_tokens: np.ndarray, all_labels: np.ndarray, 
                               rng_key: jax.random.PRNGKey, batch_size: int):
    """
    A simple Python generator function that handles shuffling, batching, and
    converting NumPy arrays to JAX arrays.
    """
    num_examples = all_tokens.shape[0]
    num_batches = num_examples // batch_size
    
    # Create the initial range of indices for the entire dataset
    indices = np.arange(num_examples)

    # Use itertools.cycle to create an infinite loop over the epoch generator
    for epoch in itertools.cycle(range(1)): 
        
        # 1. SHUFFLE: Update the JAX key for the new epoch's shuffle
        rng_key, shuffle_key = jax.random.split(rng_key)
        
        # ⭐️ FIXED: Use hashlib to create a stable, safe 32-bit seed from the JAX key.
        # This keeps the JAX key management purely for its state and uses a safe 
        # side-effect mechanism for NumPy's PRNG.
        key_bytes = shuffle_key.tobytes()
        seed = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], byteorder='little')
        
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        # 2. BATCH: Iterate over the shuffled indices in chunks (batches)
        for i in range(num_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            
            # Use NumPy indexing to quickly slice the batch data
            batch_tokens_np = all_tokens[batch_indices]
            batch_labels_np = all_labels[batch_indices]
            
            # 3. YIELD: Convert to JAX arrays and yield the batch dictionary
            yield {
                'tokens': jnp.asarray(batch_tokens_np),
                'labels': jnp.asarray(batch_labels_np)
            }
        
        tqdm.write("--- Epoch Finished, Reshuffling Data ---")


# ----------------------------------------------------------------------
#                           MAIN EXECUTION (DATA ONLY)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # 1. SETUP
    rng = jax.random.PRNGKey(0)
    
    # 2. LOAD THE DATA
    data_path = 'cifar10_tokens.npz' 
    all_tokens, all_labels = load_numpy_data(data_path)
    
    # 3. INITIALIZE DATA ITERATOR
    rng, data_rng = jax.random.split(rng)
    
    data_iterator = create_numpy_data_iterator(
        all_tokens=all_tokens, 
        all_labels=all_labels,
        rng_key=data_rng,
        batch_size=BATCH_SIZE
    )

    print("\nData iterator (custom generator) successfully initialized. Testing stream...")
    print("------------------------------------")
    
    # 4. STREAM AND INSPECT A FEW BATCHES
    for i in tqdm(range(5), desc="Streaming Test"):
        try:
            batch = next(data_iterator)
            
            print(f"\nBatch {i+1} Extracted:")
            print(f"  Tokens shape: {batch['tokens'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Sample Tokens (first 10): {batch['tokens'][0][:10]}...")
            print(f"  Sample Labels (first 5): {batch['labels'][:5]}")
            
        except StopIteration:
            print("End of data reached.")
            break
    
    print("------------------------------------")
    print("Data streaming test complete.")