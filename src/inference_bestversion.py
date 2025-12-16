import os

# --- MEMORY SETTINGS ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# Limit memory usage to avoid OOM crashes while debugging
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90" 

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
import gc

# Flax utilities
from flax.training import checkpoints
from flax.training import train_state
from flax import nnx
import orbax.checkpoint as ocp
from typing import Optional, TypedDict, Any, TypeVar, Generic
from abc import ABC, abstractmethod

# Local imports
import maskgit_class_cond_config as config
import maskgit_transformers as transformer
import parallell_decode

# ==============================================================================
# ========================== 1. CONFIGURATIONS =================================
# ==============================================================================

class Config(TypedDict):
    id: str
    codebook_size: Optional[int]
    codebook_shape: Optional[list[int]]
    embedding_dim: Optional[int]
    commitment_cost: float
    entropy_loss_type: str
    entropy_loss_ratio: float
    entropy_temperature: float
    filters: int
    num_res_blocks: int
    channel_multipliers: list[int]
    norm_type: str
    quantizer_type: str
    image_channels: int

fsqvae_config = Config({
    "id": "fsqvae_9x9x9", 
    "codebook_size": 729, 
    "embedding_dim": None,
    "codebook_shape": [9, 9, 9], 
    "commitment_cost": 0.25,
    "entropy_loss_type": "softmax",
    "entropy_loss_ratio": 0.0,
    "entropy_temperature": 1.0,
    "filters": 128,
    "num_res_blocks": 2,
    "channel_multipliers": [1, 2, 4],
    "norm_type": "LN",
    "quantizer_type": "fsq",
    "image_channels": 3,
})

vqvae_config = Config({
    "id": "vqvae", 
    "codebook_size": 512,
    "embedding_dim": 64, 
    "codebook_shape": None,
    "commitment_cost": 0.25,
    "entropy_loss_type": "softmax",
    "entropy_loss_ratio": 0.1,
    "entropy_temperature": 0.01,
    "filters": 128,
    "num_res_blocks": 2,
    "channel_multipliers": [1, 2, 4],
    "norm_type": "GN",
    "quantizer_type": "standard", 
    "image_channels": 3,
})

# ==============================================================================
# ========================== 2. MODEL DEFINITIONS ==============================
# ==============================================================================

# --- Helpers ---
def get_norm_layer(norm_type: str, num_features: int, rngs: nnx.Rngs):
    if norm_type == 'LN': return nnx.LayerNorm(num_features=num_features, rngs=rngs)
    elif norm_type == 'GN': return nnx.GroupNorm(num_features=num_features, rngs=rngs)
    else: raise NotImplementedError(f"Norm type {norm_type} not implemented")

def upsample_2d(x: jax.Array, factor: int = 2) -> jax.Array:
    n, h, w, c = x.shape
    return jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')

class ResBlock(nnx.Module):
    def __init__(self, in_features, out_features, norm_type, activation_fn, rngs):
        self.norm_1 = get_norm_layer(norm_type, in_features, rngs)
        self.activation_fn1 = activation_fn
        self.conv_1 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(3,3), use_bias=False, padding='SAME', rngs=rngs)
        self.norm_2 = get_norm_layer(norm_type, out_features, rngs)
        self.activation_fn2 = activation_fn
        self.conv_2 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=(3,3), use_bias=False, padding='SAME', rngs=rngs)
        self.residual_conv = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(1,1), use_bias=False, rngs=rngs)

    def __call__(self, x):
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

def multi_resblock(num_blocks, in_f, out_f, norm_type, act_fn, rngs):
    return nnx.Sequential(*[ResBlock(in_f if i==0 else out_f, out_f, norm_type, act_fn, rngs) for i in range(num_blocks)])

class Decoder(nnx.Module):
    def __init__(self, embedding_dim, filters, num_res_blocks, channel_multipliers, image_channels, norm_type, rngs, activation_fn=nnx.swish):
        self.norm_layer = get_norm_layer(norm_type, filters, rngs)
        self.activation_fn = activation_fn
        self.initial_conv = nnx.Conv(in_features=embedding_dim, out_features=filters*channel_multipliers[-1], kernel_size=(3,3), use_bias=True,padding='SAME', rngs=rngs)
        
        self.initial_res_blocks = multi_resblock(num_res_blocks, filters*channel_multipliers[-1], filters*channel_multipliers[-1], norm_type, activation_fn, rngs)
        
        layers = []
        for i in reversed(range(len(channel_multipliers))):
            layers.append(multi_resblock(num_res_blocks, filters*channel_multipliers[i], filters*channel_multipliers[i-1] if i > 0 else filters, norm_type, activation_fn, rngs))
            if i > 0:
                layers.append(lambda x: upsample_2d(x, 2))
                layers.append(nnx.Conv(in_features=filters*channel_multipliers[i-1] if i > 0 else filters, 
                out_features=filters*channel_multipliers[i-1] if i > 0 else filters, kernel_size=(3,3), rngs=rngs))
        
        self.decoder_blocks = nnx.Sequential(*layers)
        self.final_conv = nnx.Conv(in_features=filters, out_features=image_channels, kernel_size=(3,3), padding='SAME', rngs=rngs)

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

# --- FSQ Implementation (WITH FRIEND'S LOGIC + NORMALIZATION) ---
class FiniteScalarQuantizer(nnx.Module, Quantizer[None]):
    def __init__(self, L: list[int]):
        self.L = jnp.array(L, dtype=jnp.int32)
    
    def __call__(self, x): return x, None

    def decode_from_idx(self, ids: jax.Array) -> jax.Array:
        levels = self.L
        strides = jnp.concatenate(
            [jnp.cumprod(levels[::-1])[::-1][1:], jnp.array([1])]
        )
        idx = ids[..., None]
        shifted = (idx // strides) % levels
        offsets = levels // 2
        codes = shifted - offsets # Range approx [-4, 4]
        
        # Normalization [-1, 1]
        normalized = codes / offsets
        return normalized.astype(jnp.float32)

# --- VQ Implementation ---
class VectorQuantizerCodebook(nnx.Param[jax.Array]): pass

class VectorQuantizer(nnx.Module, Quantizer[Any]):
    def __init__(self, codebook_size, embedding_dim, rngs, **kwargs):
        init_fn = nnx.initializers.variance_scaling(scale=1.0, mode="fan_out", distribution="uniform")
        self.codebook = VectorQuantizerCodebook(init_fn(rngs.params(), shape=(codebook_size, embedding_dim)))

    def __call__(self, x): return x, {} 

    def decode_from_idx(self, ids: jax.Array) -> jax.Array:
        return jnp.take(self.codebook[...], ids, axis=0)

# --- VQVAE Container ---
class Encoder(nnx.Module):
    def __init__(self, *args, **kwargs): pass
    def __call__(self, x): return x

class VQVAE(nnx.Module, Generic[T]):
    def __init__(self, config: dict, quantizer: Quantizer[T], rngs: nnx.Rngs):
        emb_dim = len(config['codebook_shape']) if config['quantizer_type'] == 'fsq' else config['embedding_dim']
        self.quantizer = quantizer
        self.encoder = Encoder()
        self.decoder = Decoder(
            embedding_dim=emb_dim, filters=config['filters'], num_res_blocks=config['num_res_blocks'],
            channel_multipliers=config['channel_multipliers'], image_channels=config['image_channels'],
            norm_type=config['norm_type'], rngs=rngs
        )

    def decode_from_indices(self, z_ids: jax.Array) -> jax.Array:
        z_vectors = self.quantizer.decode_from_idx(z_ids)
        return self.decoder(z_vectors)

# ==============================================================================
# ========================== 3. LOADING UTILS ==================================
# ==============================================================================

def create_fsq_model(config: Config, rngs: nnx.Rngs) -> VQVAE[None]:
    quantizer = FiniteScalarQuantizer(L = config['codebook_shape'])
    return VQVAE(config=config, quantizer=quantizer, rngs=rngs)

def create_vq_model(config: Config, rngs: nnx.Rngs) -> VQVAE[Any]:
    quantizer = VectorQuantizer(codebook_size=config['codebook_size'], embedding_dim=config['embedding_dim'], rngs=rngs)
    return VQVAE(config=config, quantizer=quantizer, rngs=rngs)

def load_stage1_checkpoint(model_dir: str, config: Config, create_fn):
    model_dir = os.path.abspath(model_dir)
    options = ocp.CheckpointManagerOptions(create=False) 
    manager = ocp.CheckpointManager(os.path.join(model_dir, "named", config["id"]), options=options)
    
    concrete_model = create_fn(config, nnx.Rngs(0))
    graphdef, state = nnx.split(concrete_model)
    
    step = manager.latest_step()
    if step is not None:
        print(f"Loading {config['id']} from step {step}...")
        restored_state = manager.restore(step, args=ocp.args.PyTreeRestore(item=state, partial_restore=True))
        return nnx.merge(graphdef, restored_state)
    else:
        print(f"ERROR: No checkpoint found for {config['id']} in {manager.directory}")
        return None

def find_sequence_length_in_params(params):
    """
    Search for positional embedding to determine the true training sequence length.
    """
    # Common Flax Transformer keys for pos embedding
    keys_to_check = ['pos_embed', 'posembed_input', 'position_embedding']
    
    # 1. Flatten params dictionary
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    
    for path, value in flat_params:
        path_str = '/'.join([str(p.key) for p in path])
        # Check if this parameter looks like positional embedding
        if any(k in path_str for k in keys_to_check):
            # Shape is usually (1, SeqLen, Hidden) or (SeqLen, Hidden)
            shape = value.shape
            if len(shape) >= 2:
                # The sequence length is usually the second to last dimension (or largest dim)
                seq_len = shape[-2] if shape[0] == 1 else shape[0]
                print(f"FOUND POSITIONAL EMBEDDING: {path_str} with shape {shape}. Detected Seq Len: {seq_len}")
                return seq_len
    
    print("WARNING: Could not find positional embedding in params. Defaulting to 256.")
    return 256

def load_maskgit_checkpoint(checkpoint_dir, codebook_size):
    print(f"Loading MaskGIT from: {checkpoint_dir}")
    config_obj = config.get_config()
    
    # We initialize with a safe default, then RE-INITIALIZE after finding true shape
    temp_seq_len = 256 
    
    model = transformer.Transformer(
        vocab_size=codebook_size + 1,
        num_classes=config_obj.num_class,
        hidden_size=config_obj.transformer.num_embeds, num_hidden_layers=config_obj.transformer.num_layers,
        num_attention_heads=config_obj.transformer.num_heads, intermediate_size=config_obj.transformer.intermediate_size,
        hidden_dropout_prob=config_obj.transformer.dropout_rate
    )
    
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((1, temp_seq_len), dtype=jnp.int32)
    dummy_y = jnp.ones((1,), dtype=jnp.int32)
    initial_params = model.init(rng, dummy_x, dummy_y, deterministic=True)['params']
    
    initial_state = train_state.TrainState.create(apply_fn=model.apply, params=initial_params, tx=optax.adamw(1e-3))
    
    # Load Weights
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    loaded_state = initial_state
    
    is_orbax = False
    if os.path.exists(os.path.join(checkpoint_dir, "manifest.ocdbt")) or os.path.exists(os.path.join(checkpoint_dir, "checkpoint")):
        try:
            mgr = ocp.CheckpointManager(checkpoint_dir, options=ocp.CheckpointManagerOptions(create=False))
            step = mgr.latest_step()
            if step is not None:
                # We use PyTreeRestore to allow partial loading if shapes mistmatch, 
                # but we ideally want to inspect the loaded shape
                loaded_state = mgr.restore(step, args=ocp.args.StandardRestore(initial_state))
                is_orbax = True
        except Exception as e:
            print(f"Orbax restore attempt failed: {e}")

    if not is_orbax:
        loaded_state = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=initial_state, prefix="maskgit_")
        
    if loaded_state is initial_state:
        raise FileNotFoundError(f"Could not restore any checkpoint from {checkpoint_dir}")

    # === DYNAMIC LENGTH CORRECTION ===
    true_seq_len = find_sequence_length_in_params(loaded_state.params)
    print(f"MaskGIT Model loaded. Using detected sequence length: {true_seq_len}")
    
    del initial_state
    jax.clear_caches()
    return model, loaded_state.params, config_obj, true_seq_len

# ========================== 4. INFERENCE PIPELINE =============================

def save_images(images, prefix):
    from PIL import Image
    for i in range(images.shape[0]):
        arr = np.array(images[i])
        arr = (arr * 255).astype(np.uint8)
        fname = f"{prefix}_sample_{i}.png"
        Image.fromarray(arr).save(fname)
        print(f"Saved {fname}")

def sanity_check_decoder(pipeline_name, stage1_model, codebook_size, batch_size=1):
    """
    Generates random latent codes and decodes them.
    If this produces colorful static, the VAE works.
    If this produces white/grey blur, the VAE is broken.
    """
    print(f"\n[{pipeline_name}] Running VAE Sanity Check (Random Noise)...")
    
    # We need to know the shape. Try 16x16 (256 seq) first as it's standard
    h = w = 8 
    
    # Generate random indices
    rng = jax.random.PRNGKey(42)
    random_indices = jax.random.randint(rng, (batch_size, h, w), 0, codebook_size)
    
    @nnx.jit
    def decode(m, i): return m.decode_from_indices(i)

    images = decode(stage1_model, random_indices)
    
    # Normalize
    images = (images + 1.0) / 2.0
    images = jnp.clip(images, 0.0, 1.0)
    
    save_images(images, f"SANITY_CHECK_{pipeline_name}")
    print("Sanity Check Done. Check the file. If it's static/noise, VAE is good.")


def generate_pipeline(
    pipeline_name: str,
    stage1_model,
    maskgit_model,
    maskgit_params,
    maskgit_config,
    target_class,
    num_samples,
    num_steps,
    mask_token_id,
    seq_length # Passed from loader
):
    print(f"\n=== Running {pipeline_name} Generation Pipeline ===")
    
    EXPECTED_SEQ_LEN = 64
    if seq_length != EXPECTED_SEQ_LEN:
        print(f"WARNING: MaskGIT checkpoint's detected sequence length is {seq_length}. ")
        print(f"Using **{EXPECTED_SEQ_LEN}** (8x8 latent map) to generate a 32x32 image, as implied by VAE config.")
        seq_length = EXPECTED_SEQ_LEN # Force correct sequence length
    
    class_labels = jnp.full((num_samples,), target_class, dtype=jnp.int32) 
    initial_tokens = jnp.full((num_samples, seq_length), mask_token_id, dtype=jnp.int32)
    
    print(f"Generating with Seq Length: {seq_length}, Mask ID: {mask_token_id}")

    def tokens_to_logits(token_ids):
        return maskgit_model.apply(
            {'params': maskgit_params}, input_ids=token_ids, class_labels=class_labels, deterministic=True
        )
    tokens_to_logits_jit = jax.jit(tokens_to_logits)

    # 2. Run Parallel Decoding
    print(f"Generating tokens ({num_steps} steps)...")
    rng = jax.random.PRNGKey(int(time.time()))
    t0 = time.time()
    
    final_sequences = parallell_decode.decode(
        inputs=initial_tokens, rng=rng, tokens_to_logits=tokens_to_logits_jit,
        mask_token_id=mask_token_id, 
        num_iter=num_steps, start_iter=0,
        choice_temperature=maskgit_config.sample_choice_temperature,
        mask_scheduling_method=maskgit_config.mask_scheduling_method
    )
    final_sequences.block_until_ready()
    print(f"Token generation done: {time.time()-t0:.2f}s")
    
    generated_tokens = final_sequences[:, -1, :] 
    
    # DEBUG: Check if we have only mask tokens or zeros
    print(f"Debug Tokens - Min: {jnp.min(generated_tokens)}, Max: {jnp.max(generated_tokens)}")
    if jnp.all(generated_tokens == 0):
        print("CRITICAL WARNING: MaskGIT generated all ZEROS. The image will be a solid blob.")
        
    # 3. Decode Tokens using Stage 1 Model
    print(f"Decoding tokens to pixels using {pipeline_name} Decoder...")
    b, l = generated_tokens.shape
    h = w = int(np.sqrt(l))
    if h * w != l:
        # This will be triggered if the forced length (64) doesn't match the MaskGIT output shape.
        # We rely on the fix above (seq_length = 64) to make this 8x8.
        print(f"ERROR: Sequence length {l} is not a perfect square. Using assumed 8x8 latent.")
        h = w = 8
    print(f"Reshaping latents to {h}x{w}")
    latent_indices = generated_tokens.reshape(b, h, w)
    
    @nnx.jit
    def decode(m, i): return m.decode_from_indices(i)
    
    images = decode(stage1_model, latent_indices)
    
    print(f"Raw Decoder Output - Min: {jnp.min(images):.2f}, Max: {jnp.max(images):.2f}")
    
    images = (images + 1.0) / 2.0 
    images = jnp.clip(images, 0.0, 1.0)
    
    return images


def run_comparison(
    target_class_id: int, 
    fsq_maskgit_path: str,
    vq_maskgit_path: str,
    models_dir: str = './models',
    batch_size: int = 1,
    num_steps: int = 12
):
    # --- PIPELINE 1: FSQ ---
    try:
        print("\n>>> INITIALIZING FSQ PIPELINE")
        fsq_codebook_size = int(np.prod(fsqvae_config['codebook_shape'])) 
        
        fsq_vae = load_stage1_checkpoint(models_dir, fsqvae_config, create_fsq_model)
        if fsq_vae:
            fsq_vae.eval()
            
            # 1. Run Sanity Check
            sanity_check_decoder("FSQ", fsq_vae, fsq_codebook_size)
            
            # 2. Load MaskGIT
            mg_model, mg_params, mg_conf, true_seq_len = load_maskgit_checkpoint(fsq_maskgit_path, fsq_codebook_size)
            
            fsq_imgs = generate_pipeline(
                "FSQ", fsq_vae, mg_model, mg_params, mg_conf, 
                target_class_id, batch_size, num_steps,
                mask_token_id=fsq_codebook_size,
                seq_length=true_seq_len
            )
            save_images(fsq_imgs, f"class_{target_class_id}_FSQ")
            
            del fsq_vae, mg_model, mg_params, fsq_imgs
            jax.clear_caches()
            gc.collect()
        else:
            print("Skipping FSQ generation (Model not found)")
    except Exception as e:
        print(f"FSQ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    # --- PIPELINE 2: VQ ---
    try:
        print("\n>>> INITIALIZING VQ PIPELINE")
        vq_codebook_size = vqvae_config['codebook_size']
        
        vq_vae = load_stage1_checkpoint(models_dir, vqvae_config, create_vq_model)
        if vq_vae:
            vq_vae.eval()
            
            sanity_check_decoder("VQ", vq_vae, vq_codebook_size)

            mg_model_vq, mg_params_vq, mg_conf_vq, true_seq_len_vq = load_maskgit_checkpoint(vq_maskgit_path, vq_codebook_size)
            
            vq_imgs = generate_pipeline(
                "VQ", vq_vae, mg_model_vq, mg_params_vq, mg_conf_vq, 
                target_class_id, batch_size, num_steps,
                mask_token_id=vq_codebook_size,
                seq_length=true_seq_len_vq
            )
            save_images(vq_imgs, f"class_{target_class_id}_VQ")
            
            del vq_vae, mg_model_vq, mg_params_vq, vq_imgs
            jax.clear_caches()
            gc.collect()
        else:
            print("Skipping VQ generation (Model not found)")
    except Exception as e:
        print(f"VQ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    MODELS_ROOT = os.path.abspath('./models')
    PATH_TO_FSQ_MASKGIT = os.path.abspath('./maskgit_30') 
    PATH_TO_VQ_MASKGIT = os.path.abspath('./maskgit_100') 

    run_comparison(
        target_class_id=4, # in cifar10
        fsq_maskgit_path=PATH_TO_FSQ_MASKGIT,
        vq_maskgit_path=PATH_TO_VQ_MASKGIT,
        models_dir=MODELS_ROOT,
        batch_size=5, 
        num_steps=12
    )