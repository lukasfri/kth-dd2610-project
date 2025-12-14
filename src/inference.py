import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import os
import time

# Flax utilities for loading the MaskGIT model
from flax.training import checkpoints
from flax.training import train_state

# --- FSQ / NNX Imports ---
from flax import nnx
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import preservation_policy as ocp_pp
from typing import Optional, Callable, TypedDict, Literal, TypeVar, Generic, Any
from abc import ABC, abstractmethod

# Import your local files
import maskgit_class_cond_config as config
import maskgit_transformers as transformer
import parallell_decode

# --- CONSTANTS DERIVED FROM TRAINING SCRIPT ---
CODEBOOK_SIZE = 512
MASK_TOKEN_ID = 512 
TRAINING_SEQ_LEN = 64

# ==============================================================================
# ========================== 1. FSQ-VAE MODEL DEFINITIONS ======================
# ==============================================================================

# Helper functions needed for FSQ-VAE
def get_norm_layer(norm_type: Literal["BN", "LN", "GN"], num_features: int, rngs: nnx.Rngs) -> Callable[[jax.Array], jax.Array]:
    if norm_type == 'LN':
        return nnx.LayerNorm(num_features=num_features, rngs=rngs)
    elif norm_type == 'GN':
        return nnx.GroupNorm(num_features=num_features, rngs=rngs)
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented")

def upsample_2d(x: jax.Array, factor: int = 2) -> jax.Array:
    n, h, w, c = x.shape
    return jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')

class ResBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, norm_type: Literal['BN', 'LN', 'GN'], activation_fn: Callable, rngs: nnx.Rngs):
        self.norm_1 = get_norm_layer(norm_type, in_features, rngs)
        self.activation_fn1 = activation_fn
        self.conv_1 = nnx.Conv(in_features, kernel_size=(3, 3), use_bias=False, out_features=out_features, rngs=rngs)
        self.norm_2 = get_norm_layer(norm_type, out_features, rngs)
        self.activation_fn2 = activation_fn
        self.conv_2 = nnx.Conv(out_features, kernel_size=(3, 3), use_bias=False, out_features=out_features, rngs=rngs)
        self.residual_conv = nnx.Conv(in_features, kernel_size=(1, 1), use_bias=False, out_features=out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.norm_1(x)
        x = self.activation_fn1(x)
        x = self.conv_1(x)
        x = self.norm_2(x)
        x = self.activation_fn2(x)
        x = self.conv_2(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = self.residual_conv(residual)
        return x + residual

def multi_resblock(num_blocks: int, in_features: int, out_features: int, norm_type: str, activation_fn: Callable, rngs: nnx.Rngs) -> nnx.Sequential:
    return nnx.Sequential(*[
        ResBlock(in_features=in_features if i == 0 else out_features, out_features=out_features, norm_type=norm_type, activation_fn=activation_fn, rngs=rngs)
        for i in range(num_blocks)
    ])

class Decoder(nnx.Module):
    def __init__(self, embedding_dim: int, filters: int, num_res_blocks: int, channel_multipliers: list[int], image_channels: int, norm_type: str, rngs: nnx.Rngs, activation_fn=nnx.swish):
        self.norm_layer = get_norm_layer(norm_type, filters, rngs)
        self.activation_fn = activation_fn
        self.initial_conv = nnx.Conv(in_features=embedding_dim, kernel_size=(3, 3), use_bias=True, out_features=filters * channel_multipliers[-1], rngs=rngs)
        
        self.initial_res_blocks = multi_resblock(num_blocks=num_res_blocks, in_features=filters * channel_multipliers[-1], out_features=filters * channel_multipliers[-1], norm_type=norm_type, activation_fn=activation_fn, rngs=rngs)
        
        layers = []
        for i in reversed(range(len(channel_multipliers))):
            layers.append(multi_resblock(
                num_blocks=num_res_blocks,
                in_features=filters * channel_multipliers[i],
                out_features=filters * channel_multipliers[i-1] if i > 0 else filters,
                norm_type=norm_type, activation_fn=activation_fn, rngs=rngs
            ))
            if i > 0:
                layers.append(lambda x: upsample_2d(x, 2))
                layers.append(nnx.Conv(
                    in_features=filters * channel_multipliers[i-1] if i > 0 else filters,
                    out_features=filters * channel_multipliers[i-1] if i > 0 else filters,
                    kernel_size=(3, 3), rngs=rngs
                ))
        self.decoder_blocks = nnx.Sequential(*layers)
        self.final_conv = nnx.Conv(in_features=filters, out_features=image_channels, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, x):
        x = self.initial_conv(x)
        x = self.initial_res_blocks(x)
        x = self.decoder_blocks(x)
        x = self.norm_layer(x)
        x = self.activation_fn(x)
        x = self.final_conv(x)
        return x

T = TypeVar('T')

class Quantizer(ABC, Generic[T]):
    @abstractmethod
    def decode_from_idx(self, ids: jax.Array) -> jax.Array:
        pass

class FiniteScalarQuantizer(nnx.Module, Quantizer[None]):
    latent_dim: int
    L: jax.Array
    
    def __init__(self, L: list[int]):
        self.latent_dim = len(L)
        self.L = jnp.array(L, dtype=jnp.int32)
        
    def q_func(self, x: jax.Array) -> jax.Array:
        L_broadcast = self.L[None, :] 
        return jnp.tanh(x) * jnp.floor(L_broadcast / 2)

    def __call__(self, z: jax.Array) -> tuple[jax.Array, None]:
        B, H, W, D = z.shape
        z_flat = z.reshape((B * H * W, D))
        z_q_flat = jnp.round(self.q_func(z_flat))
        z_q = z_q_flat.reshape((B, H, W, D))
        return z_q, None
    
    def decode_from_idx(self, ids: jax.Array) -> jax.Array:
        """
        Converts flat indices (0..CODEBOOK_SIZE-1) back to quantized vectors.
        """
        # 1. Unravel the flat index into multi-dimensional indices based on L
        coords = jnp.unravel_index(ids, self.L) # Tuple of arrays
        
        # Stack coordinates to shape [..., D]
        coords = jnp.stack(coords, axis=-1)
        
        # 2. Shift coordinates to be centered around 0 (FSQ logic)
        shifts = jnp.floor(self.L / 2)
        quantized_vectors = coords - shifts
        
        return quantized_vectors.astype(jnp.float32)

class Encoder(nnx.Module):
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class VQVAE(nnx.Module, Generic[T]):
    quantizer: Quantizer[T]
    encoder: Encoder
    decoder: Decoder

    def __init__(self, config: dict, quantizer: Quantizer[T], rngs: nnx.Rngs):
        embedding_dim = len(config['codebook_shape'])
        self.quantizer = quantizer
        self.encoder = Encoder() # Dummy encoder
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            filters=config['filters'],
            num_res_blocks=config['num_res_blocks'],
            channel_multipliers=config['channel_multipliers'],
            image_channels=config['image_channels'],
            norm_type=config['norm_type'],
            rngs=rngs
        )

    def decode_from_indices(self, z_ids: jax.Array) -> jax.Array:
        z_vectors = self.quantizer.decode_from_idx(z_ids)
        reconstructed_image = self.decoder(z_vectors)
        return reconstructed_image

# FSQ Configuration
class Config(TypedDict):
    id: str
    codebook_shape: list[int]
    filters: int
    num_res_blocks: int
    channel_multipliers: list[int]
    norm_type: str
    quantizer_type: str
    image_channels: int

fsqvae_config = Config({
    "id": "fsqvae",
    "codebook_shape": [8, 8, 8], # Produces 512 codes
    "filters": 128,
    "num_res_blocks": 2,
    "channel_multipliers": [1, 2, 4],
    "norm_type": "LN",
    "quantizer_type": "fsq",
    "image_channels": 3,
})


# ==============================================================================
# ========================== 2. MODEL LOADING LOGIC ============================
# ==============================================================================

# --- STAGE II: TRANSFORMER MODEL LOADING ---
def load_inference_model(config_obj, checkpoint_dir, step=None):
    """Loads the trained MaskGIT Transformer model parameters."""
    print(f"Loading MaskGIT Transformer model from directory: {checkpoint_dir}")
    
    model = transformer.Transformer(
        vocab_size=CODEBOOK_SIZE + 1,
        num_classes=config_obj.num_class,
        hidden_size=config_obj.transformer.num_embeds,
        num_hidden_layers=config_obj.transformer.num_layers,
        num_attention_heads=config_obj.transformer.num_heads,
        intermediate_size=config_obj.transformer.intermediate_size,
        hidden_dropout_prob=config_obj.transformer.dropout_rate
    )
    
    rng_key = jax.random.PRNGKey(0)
    dummy_tokens = jnp.ones((1, TRAINING_SEQ_LEN), dtype=jnp.int32)
    dummy_labels = jnp.ones((1,), dtype=jnp.int32)

    initial_params = model.init(rng_key, dummy_tokens, dummy_labels, deterministic=True)['params']
    
    dummy_tx = optax.adamw(learning_rate=1e-3) 
    initial_state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=dummy_tx
    )
    
    print("Attempting to restore MaskGIT parameters...")
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    loaded_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir, target=initial_state, step=step, prefix="maskgit_"
    )
    
    # Fallback to Orbax if needed
    if loaded_state is initial_state: 
         if os.path.exists(os.path.join(checkpoint_dir, "manifest.ocdbt")) or os.path.exists(os.path.join(checkpoint_dir, "checkpoint")): 
            print("Trying Orbax restore for Transformer...")
            mgr = ocp.CheckpointManager(checkpoint_dir)
            step = mgr.latest_step()
            loaded_state = mgr.restore(step, args=ocp.args.StandardRestore(initial_state))

    loaded_params = loaded_state.params
    print("MaskGIT Model loaded.")
    return model, loaded_params

# --- STAGE I: FSQ MODEL LOADING ---
def load_fsq_model(model_dir: str):
    """Loads the FSQ-VAE model using NNX and Orbax."""
    print(f"Loading FSQ-VAE model from: {model_dir}")
    
    def create_model(cfg, rngs):
        quantizer = FiniteScalarQuantizer(L=cfg['codebook_shape'])
        return VQVAE(config=cfg, quantizer=quantizer, rngs=rngs)

    # 1. Create Abstract Model
    abstract_model = nnx.eval_shape(lambda: create_model(fsqvae_config, nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)

    # 2. Setup Orbax Manager
    ckpt_path = os.path.join(model_dir, "named", fsqvae_config["id"])
    if not os.path.exists(ckpt_path):
        # Fallback to direct path
        ckpt_path = model_dir
    
    print(f"Looking for FSQ checkpoints in: {ckpt_path}")
    manager = ocp.CheckpointManager(ckpt_path)
    step_to_restore = manager.latest_step()

    if step_to_restore is None:
        raise FileNotFoundError(f"No FSQ checkpoint found in {ckpt_path}")

    # 3. Restore
    print(f"Restoring FSQ at step {step_to_restore}...")
    restored_state = manager.restore(step_to_restore, args=ocp.args.StandardRestore(abstract_state))
    
    # 4. Merge back into a runnable model
    fsq_model = nnx.merge(graphdef, restored_state)
    fsq_model.eval() 
    print("FSQ-VAE Model loaded successfully.")
    
    return fsq_model

# ==============================================================================
# ========================== 3. INFERENCE LOGIC ================================
# ==============================================================================

def decode_tokens_to_image(tokens, fsq_decoder_module):
    """
    Decodes discrete tokens [B, L] into images [B, H, W, 3] using the FSQ model.
    Handles None decoder by returning black images.
    Blocks until GPU is done to avoid hanging later.
    """
    t0 = time.time()
    
    # Reshape tokens: [B, 64] -> [B, 8, 8] assuming 8x8 latent grid
    b, l = tokens.shape
    h = w = int(np.sqrt(l)) 
    latent_indices = tokens.reshape(b, h, w)
    
    print(f"Tokens prepared for FSQ. Shape: {latent_indices.shape}")

    # --- CASE 1: No FSQ Model ---
    if fsq_decoder_module is None:
        print("No FSQ decoder provided. Generating placeholder (black) images...")
        # Assume 16x upsampling (8x8 -> 128x128)
        images = jnp.zeros((b, h * 16, w * 16, 3), dtype=jnp.float32)
        # CRITICAL: Wait for this array to exist
        images.block_until_ready()
        print(f"Placeholder generation took {time.time() - t0:.3f}s")
        return images

    # --- CASE 2: FSQ Model Present ---
    try:
        @nnx.jit
        def _decode(model, idxs):
            return model.decode_from_indices(idxs)

        print("JIT Compiling and Running FSQ Decoder...")
        images = _decode(fsq_decoder_module, latent_indices)
        
        print("Waiting for GPU to finish FSQ decoding...")
        # CRITICAL: Wait for computation
        images.block_until_ready()
        
        print(f"Output Image Shape: {images.shape}")

        # Normalize [-1, 1] -> [0, 1]
        images = (images + 1.0) / 2.0
        images = jnp.clip(images, 0.0, 1.0)
        
        print(f"Decoding done in {time.time() - t0:.3f}s")
        return images

    except Exception as e:
        print(f"FSQ Error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to black image
        fallback = jnp.zeros((b, h * 16, w * 16, 3), dtype=jnp.float32)
        fallback.block_until_ready()
        return fallback

def save_image(image, filename):
    print(f"Saving {filename}...")
    # Convert to numpy (this will be fast because we blocked previously)
    arr = np.array(image)
    
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(-1)

    from PIL import Image
    try:
        Image.fromarray(arr).save(filename)
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

def run_generation(
    target_class_id: int, 
    batch_size: int = 4, 
    num_steps: int = 12, 
    checkpoint_dir: str = './checkpoints4',
    fsq_model_dir: str = './models'
):
    """Runs the scheduled parallel decoding process."""
    
    # 1. Load Config
    config_obj = config.get_config()
    
    # 2. Load MaskGIT (Stage II)
    transformer_model, trained_params = load_inference_model(config_obj, checkpoint_dir)
    
    # 3. Load FSQ-VAE (Stage I)
    try:
        fsq_model = load_fsq_model(fsq_model_dir)
    except Exception as e:
        print(f"WARNING: Could not load FSQ model: {e}")
        print("Will generate black images.")
        fsq_model = None

    # Setup Inputs
    seq_length = config_obj.image_size * config_obj.image_size // (4**2) 
    class_labels = jnp.full((batch_size,), target_class_id, dtype=jnp.int32) 
    mask_id = config_obj.transformer.mask_token_id
    initial_masked_tokens = jnp.full((batch_size, seq_length), mask_id, dtype=jnp.int32)
    
    # Define Logits Function
    def tokens_to_logits(token_ids):
        return transformer_model.apply(
            {'params': trained_params},
            input_ids=token_ids,
            class_labels=class_labels,
            deterministic=True
        )

    tokens_to_logits_jit = jax.jit(tokens_to_logits)

    # Warm-up
    print("Warming up transformer JIT...")
    try: 
        _ = tokens_to_logits_jit(initial_masked_tokens).block_until_ready()
        print("Warmup complete.")
    except Exception as e: 
        print(f"Warmup warning: {e}")

    # Run Parallel Decoding
    print(f"\nStarting Scheduled Parallel Decoding for {num_steps} steps...")
    rng_key = jax.random.PRNGKey(42)
    t_start = time.time()
    
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

    # CRITICAL: Force synchronization here
    print("MaskGIT loop queued. Waiting for GPU to finish generation...")
    final_sequences.block_until_ready()
    print(f"MaskGIT Generation finished in {time.time() - t_start:.2f}s")
    
    # Take final output
    final_tokens = final_sequences[:, -1, :] 
    
    # Decode to pixels
    final_images = decode_tokens_to_image(final_tokens, fsq_decoder_module=fsq_model) 
    
    # Save
    for i, img in enumerate(final_images):
        save_image(img, f"generated_image_class_{target_class_id}_sample_{i}.png")
    
    print("Done.")


if __name__ == '__main__':
    # Adjust paths as necessary
    run_generation(
        target_class_id=207, 
        batch_size=4, 
        num_steps=12, 
        checkpoint_dir='./maskgit_100',
        fsq_model_dir='./models'
    )