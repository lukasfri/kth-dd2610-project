import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import os
import time

# Flax utilities for loading the trained model
from flax.training import checkpoints
from flax.training import train_state

# Import your local files
import maskgit_class_cond_config as config
import maskgit_transformers as transformer
import parallell_decode

# --- CONSTANTS DERIVED FROM TRAINING SCRIPT ---
# The FSQ vocabulary size used in the training script
CODEBOOK_SIZE = 512
# The mask ID is one index past the codebook
MASK_TOKEN_ID = 512 
# The sequence length L is 64 (8x8) based on dummy inputs in create_train_state
TRAINING_SEQ_LEN = 64


# --- STAGE II: TRANSFORMER MODEL LOADING (FINAL CORRECTED VERSION) ---

def load_inference_model(config_obj, checkpoint_dir, step=None):
    """
    Loads the trained MaskGIT Transformer model parameters from a checkpoint
    saved using the flax.training.checkpoints utility.
    """
    print(f"Loading MaskGIT Transformer model from directory: {checkpoint_dir}")
    
    # 1. Define the model architecture (must match the training script exactly)
    model = transformer.Transformer(
        vocab_size=CODEBOOK_SIZE + 1, # +1 for the mask token
        num_classes=config_obj.num_class,
        hidden_size=config_obj.transformer.num_embeds,
        num_hidden_layers=config_obj.transformer.num_layers,
        num_attention_heads=config_obj.transformer.num_heads,
        intermediate_size=config_obj.transformer.intermediate_size,
        hidden_dropout_prob=config_obj.transformer.dropout_rate
    )

    print("Transformer model architecture instantiated.")

    # 2. Re-create the initial TrainState structure (the "target" for loading)
    rng_key = jax.random.PRNGKey(0)
    dummy_tokens = jnp.ones((1, TRAINING_SEQ_LEN), dtype=jnp.int32)
    dummy_labels = jnp.ones((1,), dtype=jnp.int32)

    # Initialize parameters (needed to set the structure/shapes)
    initial_params = model.init(
        rng_key, dummy_tokens, dummy_labels, deterministic=True
    )['params']
    
    # --- DEBUG STEP 1: CALCULATE HASH/SUM OF INITIAL PARAMETERS (ROBUST FIX) ---
    
    # Helper function to safely find the token embedding table sum
    def get_embedding_sum(params):
        # 1. Access the Transformer's parameters (look for top-level keys)
        transformer_params = params.get('Transformer_0', params) 

        # 2. Find the *parent* layer named for token embedding
        if 'token_embed' in transformer_params:
            parent_module_params = transformer_params['token_embed']
        elif 'Embed_0' in transformer_params:
            parent_module_params = transformer_params['Embed_0']
        else:
            try: # Fallback for a heavily nested structure
                parent_module_params = params['params']['token_embed']
            except KeyError:
                 raise KeyError("Could not find a parameter module named 'token_embed' or 'Embed_0' in the Transformer structure.")
        
        # 3. Find the *word embeddings* sub-module 
        if 'word_embeddings' in parent_module_params:
            word_embed_params = parent_module_params['word_embeddings']
        else:
            raise KeyError("Expected 'word_embeddings' key inside the embedding parent module, but not found.")

        # 4. Find the final weight tensor inside the sub-module (THE ROBUST FIX)
        final_param_key_candidates = ['kernel', 'embedding', 'w'] 
        embedding_table = word_embed_params
        
        # Iteratively try to drill down until we hit a JAX array (or fail)
        for _ in range(3): # Try up to three levels of nesting
            if not isinstance(embedding_table, dict):
                break # Found the array (or non-dict), exit loop

            # Check for candidate keys at the current level
            found_key = False
            for key in final_param_key_candidates:
                if key in embedding_table:
                    embedding_table = embedding_table[key]
                    found_key = True
                    break
            
            # If no candidate key was found, but only one key exists, assume it's the next nested module
            if not found_key and len(embedding_table) == 1:
                # Drill down into the single existing key
                embedding_table = list(embedding_table.values())[0]
                found_key = True
            
            if not found_key:
                # If we're still a dict and couldn't drill down, raise error
                available_keys = list(embedding_table.keys())
                raise KeyError(
                    f"Could not resolve the final array in 'word_embeddings'. Last checked keys: {available_keys}"
                )
        
        # Final check and return
        if hasattr(embedding_table, 'sum'): 
            return float(jnp.sum(embedding_table))
        
        raise TypeError(f"Final object is not a JAX array but a {type(embedding_table)}.")

    try:
        initial_sum = get_embedding_sum(initial_params)
    except Exception as e:
        print(f"FATAL DEBUG ERROR: Could not access initial token embedding. Error: {e}")
        # The JAX error happens inside this try/except block.
        # We raise the exception to see the custom KeyError if it triggers, 
        # or the JAX error if it fails on the final sum.
        raise e 
        
    print(f"DEBUG: Initial random embedding sum: {initial_sum:.6f}")
    
    # Create a dummy optimizer and TrainState structure
    dummy_tx = optax.adamw(learning_rate=1e-3) 
    initial_state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=dummy_tx
    )
    
    # 3. Load the checkpoint file
    print("Attempting to restore parameters...")
    
    # Normalize checkpoint directory to an absolute path (Orbax requires absolute paths)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    # First try the simple flax checkpoints loader (works if training used flax.checkpoints)
    loaded_state = None
    try:
        loaded_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target=initial_state,
            step=step,
            prefix="maskgit_",
        )
        print("flax.training.checkpoints.restore_checkpoint returned without exception.")
    except Exception as e:
        print(f"flax.restore_checkpoint failed with: {e}")

    # If flax restore didn't find a valid checkpoint (or resulted in random params),
    # detect Orbax/ocdbt checkpoint layout and try Orbax restore as a fallback.
    def _looks_like_orbax(dirpath: str) -> bool:
        # orbax checkpoints contain manifest.ocdbt or _CHECKPOINT_METADATA or ocdbt.process_*
        return (
            os.path.exists(os.path.join(dirpath, "manifest.ocdbt"))
            or os.path.exists(os.path.join(dirpath, "_CHECKPOINT_METADATA"))
            or os.path.exists(os.path.join(dirpath, "ocdbt.process_0"))
        )

    # Helper to extract params from a restored object (TrainState, dict, etc.)
    def _extract_params(restored_obj):
        # Direct attribute
        if hasattr(restored_obj, 'params'):
            return restored_obj.params
        # dict-like
        if isinstance(restored_obj, dict):
            if 'params' in restored_obj:
                return restored_obj['params']
            # sometimes nested under 'target' or similar
            if 'target' in restored_obj and isinstance(restored_obj['target'], dict) and 'params' in restored_obj['target']:
                return restored_obj['target']['params']
            # try to find any nested params key
            for v in restored_obj.values():
                try:
                    p = _extract_params(v)
                    if p is not None:
                        return p
                except Exception:
                    continue
        raise KeyError('Could not extract `params` from restored checkpoint object.')

    # Decide whether to attempt orbax restore
    try_orbax = False
    if loaded_state is None:
        try_orbax = True
    else:
        try:
            loaded_params_tmp = _extract_params(loaded_state)
            # If the sums match the initial random init, we'll treat as failed load
            try:
                loaded_sum_tmp = get_embedding_sum(loaded_params_tmp)
                if np.isclose(initial_sum, loaded_sum_tmp):
                    print("Loaded params appear identical to random init (sum match) — will try Orbax fallback if available.")
                    try_orbax = True
            except Exception:
                # Could not compute sum; still consider trying orbax if layout present
                try_orbax = try_orbax or _looks_like_orbax(checkpoint_dir)
        except Exception:
            try_orbax = True

    if try_orbax and _looks_like_orbax(checkpoint_dir):
        print("Detected Orbax/ocdbt checkpoint layout — attempting Orbax restore...")
        try:
            import orbax.checkpoint as ocp
        except Exception as e:
            print(f"Failed to import orbax.checkpoint: {e}")
            raise

        # Create a manager and restore the latest step
        try:
            manager = ocp.CheckpointManager(checkpoint_dir)
            step_to_restore = manager.latest_step()
            if step_to_restore is None:
                raise FileNotFoundError(f"No checkpoint step found in {checkpoint_dir}")

            print(f"Restoring Orbax checkpoint at step {step_to_restore} from {checkpoint_dir}...")
            restored = manager.restore(step_to_restore, args=ocp.args.StandardRestore(initial_state))
            loaded_state = restored
            print("Orbax restore completed (returned object assigned to loaded_state).")
        except Exception as e:
            print(f"Orbax restore failed: {e}")
            # propagate original exception if nothing worked
            if loaded_state is None:
                raise
    
    # 4. Extract the necessary components and perform the check
    loaded_params = loaded_state.params
    
    # --- DEBUG STEP 2 & 3: CALCULATE AND COMPARE HASH/SUM OF LOADED PARAMETERS ---
    # This also uses the robust get_embedding_sum
    loaded_sum = get_embedding_sum(loaded_params)
    
    # Compare the sums
    if np.isclose(initial_sum, loaded_sum):
        print("🛑 ---------------------------------------------------------------------------------")
        print("🛑 WARNING: Checkpoint load FAILED. Parameters are RANDOM or CHECKPOINT IS MISSING.")
        print("🛑 Sums are the same. Please verify `checkpoint_dir` contains the trained files.")
        print("🛑 ---------------------------------------------------------------------------------")
    else:
        print("✅ Checkpoint load SUCCESS.")
        print(f"DEBUG: Loaded embedding sum: {loaded_sum:.6f} (Different from initial)")
    
    print("Model parameters initialized (or loaded) and JAX structure confirmed.")
    
    return model, loaded_params


# --- STAGE I: FSQ DECODER PLACEHOLDER (Needs implementation later) ---

def decode_tokens_to_image(tokens, fsq_decoder_module):
    """
    PLACEHOLDER: This function requires the FSQ Decoder (Stage I) model 
    and parameters to convert discrete tokens into image pixels.
    """
    # Reshape the tokens from (B, L) to (B, H, W)
    h = w = int(jnp.sqrt(tokens.shape[-1]))
    latent_indices = tokens.reshape(tokens.shape[0], h, w)
    
    # NOTE: This is where you would call the FSQ Decoder model.apply(...)
    
    print(f"Tokens decoded. Final shape: {latent_indices.shape}. Ready for FSQ Decoder.")
    # Returns a black placeholder image array
    print("Starting FSQ decoding of tokens...")
    t0 = time.time()

    # If no FSQ decoder supplied, return placeholder images and log time.
    if fsq_decoder_module is None:
        out = jnp.zeros((tokens.shape[0], h * 16, w * 16, 3), dtype=jnp.float32)
        print(f"No FSQ decoder provided — returning placeholder images. Took {time.time() - t0:.3f}s")
        return out

    # If a decoder is provided, expect either:
    #  - a tuple (model, params)
    #  - or an object with `.apply(params, inputs)` behavior
    try:
        if isinstance(fsq_decoder_module, tuple) and len(fsq_decoder_module) == 2:
            model, params = fsq_decoder_module
            seq_len = tokens.shape[-1]
            side = int(np.sqrt(seq_len))
            token_inputs = tokens.reshape((tokens.shape[0], side, side)) if tokens.ndim == 2 else tokens

            # Prefer a dedicated `decode` method if present
            if hasattr(model, 'decode'):
                images = model.apply({'params': params}, token_inputs, method=model.decode)
            else:
                images = model.apply({'params': params}, token_inputs)

            images = jnp.array(images, dtype=jnp.float32)
            print(f"FSQ decoding completed in {time.time() - t0:.3f}s")
            return images
        else:
            # Try generic apply
            seq_len = tokens.shape[-1]
            side = int(np.sqrt(seq_len))
            token_inputs = tokens.reshape((tokens.shape[0], side, side)) if tokens.ndim == 2 else tokens
            images = fsq_decoder_module.apply({'params': fsq_decoder_module.params}, token_inputs)
            images = jnp.array(images, dtype=jnp.float32)
            print(f"FSQ decoding completed in {time.time() - t0:.3f}s")
            return images
    except Exception as e:
        print(f"FSQ decoding failed: {e}")
        out = jnp.zeros((tokens.shape[0], h * 16, w * 16, 3), dtype=jnp.float32)
        print(f"Returning placeholder images. Took {time.time() - t0:.3f}s")
        return out

def save_image(image, filename):
    """
    PLACEHOLDER: Saves the resulting JAX array as an image file (e.g., PNG).
    """
    print(f"Saving generated image to {filename}...")
    try:
        # Convert to numpy
        arr = np.array(image)
        # Handle float arrays in [0,1]
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        # Remove channel dim if single-channel
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(-1)

        # Lazy import of PIL or fallback to imageio
        try:
            from PIL import Image
            Image.fromarray(arr).save(filename)
        except Exception:
            try:
                import imageio
                imageio.imwrite(filename, arr)
            except Exception as e:
                print(f"Failed to save image to {filename}: {e}")
    except Exception as e:
        print(f"save_image() error for {filename}: {e}")
    
# -----------------------------------------------------------------------------

def run_generation(
    target_class_id: int, 
    batch_size: int = 4, 
    num_steps: int = 12, 
    checkpoint_dir: str = './checkpoints4'
):
    """Runs the scheduled parallel decoding process."""
    
    # Load configuration
    config_obj = config.get_config()
    
    # Load stuff
    transformer_model, trained_params = load_inference_model(config_obj, checkpoint_dir)
    
    # Calculate sequence length (e.g., 64 for CIFAR-10)
    seq_length = config_obj.image_size * config_obj.image_size // (4**2)
    # Assuming 4x downsampling appaerently, thats what the AI said
    # --- Setup Inputs ---
    
    # (B,) array with the class ID
    # batch size is how many images we want
    class_labels = jnp.full((batch_size,), target_class_id, dtype=jnp.int32) 
    
    # Initial input: an array of all masked tokens (B, L)
    # we get the ID of were predictions are to be made
    # from 3.2 in MaskGit paper
    # Initial input: an array of all masked tokens (B, L)
    mask_id = config_obj.transformer.mask_token_id
    # Set intial tokens to the mask_ID, maskID is this a value that means that it hasnt been predicted yet
    initial_masked_tokens = jnp.full((batch_size, seq_length), mask_id, dtype=jnp.int32)
    
    # --- Define the Logits Function (The heart of the inference loop) ---
    def tokens_to_logits(token_ids):
        """Feeds the current token state into the trained Transformer to get logits."""
        # Ensure the class labels match the batch size of the token_ids
        # This function runs inside the JIT-compiled decoding loop
        # we input the current state of the sequence, we use all the existing tokens to predict the masked ones
        logits = transformer_model.apply(
            {'params': trained_params},
            input_ids=token_ids,
            class_labels=class_labels,
            deterministic=True
        )
        # Logits are unnormalized prediction scores (raw numbers) for every possible token at every position in the input sequence.
        # shape [B, L, K] K is codebook size
        return logits
    # JIT-compile the logits function to avoid recompilation on every decoder
    # call. The first call will incur XLA compilation cost, but subsequent
    # calls will reuse the compiled executable and be much faster.
    tokens_to_logits_jit = jax.jit(tokens_to_logits)
    # Warm-up / compile once on a representative input to avoid a compile
    # happening inside the decoding loop (which can cause extra latency).
    try:
        print("Warming up transformer (one-time XLA compile)...")
        _ = tokens_to_logits_jit(initial_masked_tokens)
        print("Warmup complete.")
    except Exception as e:
        print(f"Warning: warmup compile failed: {e}")
    # --- Run Parallel Decoding ---
    
    rng_key = jax.random.PRNGKey(42)
    
    print(f"\nStarting Scheduled Parallel Decoding for {num_steps} steps...")
    print("--- JAX COMPILATION FOR DECODING WILL OCCUR NOW ---")

    # Call the core decoding function
    final_sequences = parallell_decode.decode(
        inputs=initial_masked_tokens,
        rng=rng_key,
        tokens_to_logits=tokens_to_logits_jit,
        mask_token_id=mask_id,
        num_iter=num_steps,
        start_iter=0,
        choice_temperature=config_obj.sample_choice_temperature,
        mask_scheduling_method=config_obj.mask_scheduling_method
    )

    print("Decoding complete.")
    
    # --- Post-Processing and FSQ Decoding ---
    
    # Take the final output (from the last iteration)
    final_tokens = final_sequences[:, -1, :] 
    
    # Convert final discrete tokens back into continuous image pixels
    final_images = decode_tokens_to_image(final_tokens, fsq_decoder_module=None) 
    
    # Save the results
    for i, img in enumerate(final_images):
        save_image(img, f"generated_image_class_{target_class_id}_sample_{i}.png")
    
    print("Generation process finished.")


if __name__ == '__main__':
    # ------------------ CODE GRAVEYARD: FAST TESTING CONFIG ------------------
    # The tiny config used for fast testing is preserved here for debugging
    # and reference. It is intentionally NOT applied so inference uses the
    # real training configuration returned by `maskgit_class_cond_config.get_config()`.
# TINY_CONFIG = ml_collections.ConfigDict({
#         'image_size': 16,     # Smallest possible image size
#         'num_class': 1000,    # Default number of classes (not changed)
#         'transformer': {
#             'mask_token_id': 257, # Visual codebook size + 1 (e.g., 256 + 1)
#             'num_embeds': 4,      # Tiny hidden dimension (was 512/768)
#             'num_layers': 1,      # Single layer Transformer (was 12/24)
#             'num_heads': 1,       # Single attention head
#             'intermediate_size': 8, # Tiny feed-forward size
#             'dropout_rate': 0.0,
#         },
#         'sample_choice_temperature': 0.0,
#         'mask_scheduling_method': 'cosine',
#     })
    
#     # Override the config object function to return the tiny config for testing
#     def get_tiny_config():
#         return TINY_CONFIG
    
#     # Temporarily replace the real config getter with the tiny one
#     config.get_config = get_tiny_config
#     # ------------------ ADD END ------------------
    
#     # Example usage: Generate 4 images of class ID 5, using 16 refinement steps.
#     # ... (rest of the run_generation call)
#     run_generation(
#         target_class_id=5, 
#         batch_size=4, 
#         num_steps=5, # Reduce steps for faster testing
#         checkpoint_path='./your_best_maskgit_checkpoint'
#     )
    # ----------------------------------------------------------------

    # Example usage: 
    # checkpoint_dir is now explicitly set to the name of the folder containing the checkpoint.
    run_generation(
        target_class_id=5, 
        batch_size=4, 
        num_steps=5, # Keep steps low for fast testing
        # FIX: Setting the checkpoint_dir to the folder name you specified.
        checkpoint_dir='./maskgit_100' 
    )