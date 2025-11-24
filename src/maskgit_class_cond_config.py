import ml_collections

def get_config():
  """Get the default hyperparameter configuration."""
  # Initialize a fresh ConfigDict (replaces base_config)
  config = ml_collections.ConfigDict()
  
  config.experiment = "maskgit_class_cond"
  config.model_class = "maskgit_class_cond"
  config.sequence_order = "horizontal"

  # Dataset settings
  config.num_class = 10    # Set this to 10 for CIFAR, 1000 for ImageNet
  config.batch_size = 256
  config.image_size = 256  # Match your FSQ latent size (e.g., 16*16 latent -> 256px input?) 
                           # NOTE: If working on Latents, this is just reference.
  
  # Optimization
  config.num_train_steps = 2_000_000
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.lr = 0.0001
  config.optimizer.beta1 = 0.9
  config.optimizer.beta2 = 0.96
  config.optimizer.warmup_steps = 5000
  config.optimizer.weight_decay = 4.5e-2

  # Loss & Masking
  config.compute_loss_for_all = False
  config.label_smoothing = 0.1
  config.mask_scheduling_method = "cosine"
  config.sample_choice_temperature = 4.5
  config.min_masking_rate = 0.5 # FSQ paper recommends 0.45 or 0.5

  # Transformer Architecture
  config.transformer = ml_collections.ConfigDict()
  config.transformer.num_layers = 12      # Reduced from 24 for stability/speed
  config.transformer.patch_size = 1       # Since we are using tokens, usually patch is 1
  config.transformer.num_embeds = 768     # Hidden dimension
  config.transformer.intermediate_size = 3072
  config.transformer.num_heads = 12
  config.transformer.dropout_rate = 0.1
  config.transformer.mask_token_id = 1000 # Matches FSQ codebook size
  
  # These are usually not needed for Stage 2 training but good to keep empty
  config.vqgan = ml_collections.ConfigDict()
  config.vqvae = ml_collections.ConfigDict()

  return config