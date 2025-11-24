# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common layers and blocks."""

import functools
from typing import Any, Tuple

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


def _l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.

  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def get_norm_layer(train, dtype, norm_type='BN'):
  """Normalization layer."""
  if norm_type == 'BN':
    norm_fn = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name=None,
        axis_index_groups=None,
        dtype=jnp.float32)
  elif norm_type == 'LN':
    norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
  elif norm_type == 'GN':
    norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
  else:
    raise NotImplementedError
  return norm_fn


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.

  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
      (1,) + strides + (1,), padding)
  return pool_sum / pool_denom


def upsample(x, factor=2):
  n, h, w, c = x.shape
  x = jax.image.resize(x, (n, h * factor, w * factor, c), method='nearest')
  return x


def dsample(x):
  return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding='same')


# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common losses used in training GANs and masked modeling."""
from typing import Any, Optional
import jax
import jax.numpy as jnp





def apply_label_smoothing(one_hot_targets: jnp.ndarray,
                          label_smoothing: float) -> jnp.ndarray:
  """Applies label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html


  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets

def weighted_sequence_cross_entropy_loss(
    *,
    labels: jnp.ndarray,
    logits: jnp.ndarray,
    weights: jnp.ndarray,
    label_smoothing: Optional[float] = 0.0):
  """Computes the mean cross-entropy for the sequence predictions.

  Args:
    labels: 2D int array of shape (B, T) where each value is in [0, C-1].
    logits: 3D array of shape (B, T, C) where C is number of classes.
    weights: 2D float array (B, T).
    label_smoothing: float.

  Returns:
    float loss.
  """
  vocab_size = logits.shape[-1]
  one_hot_targets = jax.nn.one_hot(labels, vocab_size)
  soft_targets = apply_label_smoothing(one_hot_targets, label_smoothing)
  loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)
  loss = jnp.sum(loss * weights, axis=-1) / jnp.sum(weights, axis=-1)
  return jnp.mean(loss)




# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mask schedule functions R."""
import math
import jax
import jax.numpy as jnp


def schedule(ratio, total_unknown, method="cosine"):
  """Generates a mask rate by scheduling mask functions R.

  Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
  training, the input ratio is uniformly sampled; during inference, the input
  ratio is based on the step number divided by the total iteration number: t/T.
  Based on experiements, we find that masking more in training helps.
  Args:
    ratio: The uniformly sampled ratio [0, 1) as input.
    total_unknown: The total number of tokens that can be masked out. For
      example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
      512x512 images.
    method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
      "pow2.5" represents x^2.5

  Returns:
    The mask rate (float).
  """
  if method == "uniform":
    mask_ratio = 1. - ratio
  elif "pow" in method:
    exponent = float(method.replace("pow", ""))
    mask_ratio = 1. - ratio**exponent
  elif method == "cosine":
    mask_ratio = jax.lax.cos(math.pi / 2. * ratio)
  elif method == "log":
    mask_ratio = -jnp.log2(ratio) / jnp.log2(total_unknown)
  elif method == "exp":
    mask_ratio = 1 - jnp.exp2(-jnp.log2(total_unknown) * (1 - ratio))
  # Clamps mask into [epsilon, 1)
  mask_ratio = jnp.clip(mask_ratio, 1e-6, 1.)
  return mask_ratio



# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Configuration and hyperparameter sweeps for maskgit training."""

from maskgit.configs import base_config
from maskgit.configs import vqgan_config
import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = base_config.get_config()
  config.experiment = "maskgit_class_cond"
  config.model_class = "maskgit_class_cond"
  config.sequence_order = "horizontal"

  config.num_class = 10 # was 1000, think CIFAR should want 10
  config.batch_size = 256
  config.eval_batch_size = 8
  config.eval_every_steps = 10_000
  config.checkpoint_every_steps = 10_000
  config.eval_show_num = 16
  config.image_size = 256
  config.num_train_steps = 2_000_000
  config.eval_num = 50_000
  config.eval_avg_num = 3
  config.eval_exact_match = True

  config.compute_loss_for_all = False
  config.label_smoothing = 0.1
  config.mask_scheduling_method = "cosine"
  # config.sample_num_iterations = 16
  config.sample_choice_temperature = 4.5
  config.min_masking_rate = 0.5

  config.optimizer.lr = 0.0001
  config.optimizer.beta1 = 0.9
  config.optimizer.beta2 = 0.96
  config.optimizer.warmup_steps = 5000
  config.optimizer.weight_decay = 4.5e-2

  config.transformer = ml_collections.ConfigDict()
  config.transformer.num_layers = 12 # changed from original 24
  config.transformer.patch_size = 16
  config.transformer.num_embeds = 768
  config.transformer.intermediate_size = 3072
  config.transformer.num_heads = 16
  config.transformer.dropout_rate = 0.1
  config.transformer.mask_token_id = 1000 # changed from -1
  config.transformer.latent_size = 16

  vqgan_cf = vqgan_config.get_config()
  config.vqgan = vqgan_cf.vqgan
  config.vqvae = vqgan_cf.vqvae

  return config


def get_hyper(h):
  return h.product([
      h.sweep("image_size", [256, 512]),
      h.sweep("compute_loss_for_all", [True, False]),
  ],
                   name="config")




# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""MaskGIT Transformer for masked visual token modeling (MVTM) based on BERT.

The transformer is implemented based on a simplified version of BERT
[https://arxiv.org/abs/1810.04805]. Specifically, the part on next sentence
prediction and segment ids are removed from BERT. Taking the masked tokens as
inputs, the model predicts the probability of all individual tokens.

For details, please see https://arxiv.org/abs/2012.09841.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Text, Tuple, Union

from flax import linen as nn
import jax
import jax.numpy as jnp


LAYERNORM_EPSILON = 1e-12  # Layer norm from BERT

InitializerType = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


def truncated_normal(stddev: Union[float, jnp.ndarray], dtype=jnp.float32):

  def init(key: jnp.ndarray, shape: Iterable[int], dtype: jnp.dtype = dtype):
    return jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev

  return init


class Attention(nn.Module):
  """Attention layer that is part of each Transformer layer."""
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  hidden_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input: jnp.ndarray, input_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    attention_mask = nn.make_attention_mask(input_mask, input_mask)
    attention_output = nn.attention.SelfAttention(
        num_heads=self.num_attention_heads,
        qkv_features=self.hidden_size,
        dropout_rate=self.attention_probs_dropout_prob,
        deterministic=deterministic,
        kernel_init=self.initializer_fn,
        bias_init=jax.nn.initializers.zeros,
        name='self_attention',
    )(layer_input, attention_mask)

    attention_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        attention_output, deterministic=deterministic)
    attention_output = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='attention_output_ln')(
            attention_output + layer_input)

    return attention_output


class Mlp(nn.Module):
  """MLP layer that is part of each Transformer layer."""
  hidden_size: int
  hidden_dropout_prob: float
  intermediate_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, attention_output: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    intermediate_output = nn.Dense(
        features=self.intermediate_size,
        kernel_init=self.initializer_fn,
        name='intermediate_output')(
            attention_output)
    intermediate_output = jax.nn.gelu(intermediate_output)

    layer_output = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='layer_output')(
            intermediate_output)
    layer_output = nn.Dropout(rate=self.hidden_dropout_prob)(
        layer_output, deterministic=deterministic)
    layer_output = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='layer_output_ln')(
            layer_output + attention_output)

    return layer_output


class TransformerLayer(nn.Module):
  """A single Transformer layer."""
  intermediate_size: int
  hidden_size: int
  hidden_dropout_prob: float
  num_attention_heads: int
  attention_probs_dropout_prob: float
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, layer_input: jnp.ndarray, input_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    attention_output = Attention(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        num_attention_heads=self.num_attention_heads,
        attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        initializer_fn=self.initializer_fn)(
            layer_input=layer_input,
            input_mask=input_mask,
            deterministic=deterministic)

    layer_output = Mlp(
        hidden_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        intermediate_size=self.intermediate_size,
        initializer_fn=self.initializer_fn)(
            attention_output=attention_output, deterministic=deterministic)

    return layer_output


class Embed(nn.Module):
  """Embeds visual tokens."""
  embedding_size: int
  hidden_dropout_prob: float
  vocab_size: int
  max_position_embeddings: int
  initializer_fn: InitializerType
  hidden_size: Optional[int] = None

  @nn.compact
  def __call__(self, input_ids: jnp.ndarray,
               deterministic: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    seq_length = input_ids.shape[-1]
    position_ids = jnp.arange(seq_length)[None, :]

    word_embedder = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='word_embeddings')
    word_embeddings = word_embedder(input_ids)
    position_embeddings = nn.Embed(
        num_embeddings=self.max_position_embeddings,
        features=self.embedding_size,
        embedding_init=self.initializer_fn,
        name='position_embeddings')(
            position_ids)

    input_embeddings = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='embeddings_ln')(
            word_embeddings + position_embeddings)
    if self.hidden_size:
      input_embeddings = nn.Dense(
          features=self.hidden_size,
          kernel_init=self.initializer_fn,
          name='embedding_hidden_mapping')(
              input_embeddings)
    input_embeddings = nn.Dropout(rate=self.hidden_dropout_prob)(
        input_embeddings, deterministic=deterministic)

    return input_embeddings


class Bias(nn.Module):
  """Adds a learnable bias to the input.

  Attributes:
    dtype: the dtype of the computation (default: float32).
    bias_init: initializer function for the bias.
  """
  dtype: Any = jnp.float32
  bias_init: Callable[[Any, Tuple[int], Any], Any] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)

    bias_shape = inputs.shape[-1]
    bias = self.param('bias', self.bias_init, bias_shape)
    bias = jnp.asarray(bias, self.dtype)
    bias = jnp.broadcast_to(bias, inputs.shape)

    return inputs + bias


class MlmLayer(nn.Module):
  """MLM layer for masked token prediction."""
  hidden_size: int
  initializer_fn: InitializerType

  @nn.compact
  def __call__(self, last_layer: jnp.ndarray,
               embeddings: jnp.ndarray) -> jnp.ndarray:
    mlm_hidden = nn.Dense(
        features=self.hidden_size,
        kernel_init=self.initializer_fn,
        name='mlm_dense')(
            last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = nn.LayerNorm(
        epsilon=LAYERNORM_EPSILON, name='mlm_ln')(
            mlm_hidden)
    output_weights = jnp.transpose(embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = Bias(name='mlm_bias')(logits)
    return logits


class Transformer(nn.Module):
  """Transformer modified from BERT."""
  vocab_size: int
  hidden_size: int = 768
  num_hidden_layers: int = 12
  num_attention_heads: int = 12
  intermediate_size: int = 3072
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  max_position_embeddings: int = 256
  initializer_range: float = 0.02

  @nn.compact
  def __call__(self,
               input_ids: jnp.ndarray,
               deterministic: bool = True) -> Dict[Text, jnp.ndarray]:
    input_ids = input_ids.astype('int32')
    input_embeddings = Embed(
        embedding_size=self.hidden_size,
        hidden_dropout_prob=self.hidden_dropout_prob,
        vocab_size=self.vocab_size,
        max_position_embeddings=self.max_position_embeddings,
        initializer_fn=truncated_normal(self.initializer_range))(
            input_ids=input_ids, deterministic=deterministic)

    layer_input = input_embeddings
    for _ in range(self.num_hidden_layers):
      layer_output = TransformerLayer(
          intermediate_size=self.intermediate_size,
          hidden_size=self.hidden_size,
          hidden_dropout_prob=self.hidden_dropout_prob,
          num_attention_heads=self.num_attention_heads,
          attention_probs_dropout_prob=self.attention_probs_dropout_prob,
          initializer_fn=truncated_normal(self.initializer_range))(
              layer_input=layer_input,
              input_mask=jnp.ones_like(input_ids, dtype=jnp.int32),
              deterministic=deterministic)
      layer_input = layer_output

    word_embedding_matrix = self.variables['params']['Embed_0'][
        'word_embeddings']['embedding']
    logits = MlmLayer(
        hidden_size=self.hidden_size,
        initializer_fn=truncated_normal(self.initializer_range))(
            last_layer=layer_output, embeddings=word_embedding_matrix)

    return logits
  




# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fast decoding routines for non-autoregressive generation."""

import flax
import jax
from jax import lax
import jax.numpy as jnp
from maskgit.libml import mask_schedule

# Confidence score for known tokens to avoid masking or repredicting them.
# Here we don't use 1.0 because the upper bounder of the probability can be
# possiblity larger than 1 due to the noise addition.
_CONFIDENCE_OF_KNOWN_TOKENS = jnp.inf


def mask_by_random_topk(rng, mask_len, probs, temperature=1.0):
  """Modifies from jax.random.choice without replacement.

  JAX's original implementation is as below:
    g = -gumbel(key, (n_inputs,)) - jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]
  We adds temperature annealing on top of it, which is:
    g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]

  Args:
    rng: a PRNG key used as the random key.
    mask_len: the number to mask.
    probs: the probabilities associated with each entry.
    temperature: when temperature = 1.0, it's identical to jax's implementation.
      The larger this value is, the more random the masking is picked.

  Returns:
    A binary masking map [batch_size, seq_len].
  """
  confidence = jnp.log(probs) + temperature * jax.random.gumbel(
      rng, probs.shape)
  sorted_confidence = jnp.sort(confidence, axis=-1)
  # Obtains cut off threshold given the mask lengths.
  cut_off = jnp.take_along_axis(sorted_confidence, mask_len.astype(jnp.int32), axis=-1)
  # Masks tokens with lower confidence.
  masking = (confidence < cut_off)
  return masking


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jnp.DeviceArray  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  cur_seqs: jnp.DeviceArray  # int32 [batch, seq_len]
  rng: jnp.DeviceArray  # Sampling random state.
  final_seqs: jnp.DeviceArray  # int32 [batch, num_iter, seq_len]


def state_init(init_indices, rng, num_iter, start_iter=0):
  """Initializes the decoding state data structure."""
  cur_index0 = jnp.array(start_iter)
  cur_seqs0 = init_indices
  final_seqs0 = jnp.expand_dims(init_indices, 1)
  final_seqs0 = jnp.tile(final_seqs0, (1, num_iter, 1))
  return State(
      cur_index=cur_index0, cur_seqs=cur_seqs0, rng=rng, final_seqs=final_seqs0)

def decode(inputs,
           rng,
           tokens_to_logits,
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           mask_scheduling_method="cosine"):
  """Fast decoding for iterative generation.

  Args:
    inputs: int32 array: [batch_size, seq_length] input sequence of masked
      tokens, where the masking tokens is defined by mask_token_id.
    rng: jnp.DeviceArray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and cache and
      returning logits and updated cache.
    mask_token_id: int: [Mask] token id.
    num_iter: int: default is 12.
    start_iter: int: default is 0.
    choice_temperature: float: temperature to control the randomness of masking.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.

  Returns:
     [batch_size, num_iter, seq_length] output sequence of tokens in all
       iterations.
  """
  inputs = inputs.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = state_init(inputs, rng, num_iter, start_iter=start_iter)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function."""
    rng = state.rng
    step = state.cur_index
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs

    # Calls model on current seqs to get next-iteration seqs.
    logits = tokens_to_logits(cur_ids)
    rng, sample_rng = jax.random.split(rng, 2)
    # Samples the ids using categorical sampling: [batch_size, seq_length].
    sampled_ids = jax.random.categorical(sample_rng, logits)

    # Just updates the masked tokens.
    unknown_map = (cur_ids == mask_token_id)
    sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
    # Defines the mask ratio for the next round. The number to mask out is
    # determined by mask_ratio * unknown_number_in_the_beginning.
    ratio = 1. * (step + 1) / num_iter
    mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                        mask_scheduling_method)
    # Updates final seqs with the current sampled_ids.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
    # Computes the probabilities of each selected tokens.
    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
    # Ignores the tokens given in the input by overwriting their confidence.
    selected_probs = jnp.where(unknown_map, selected_probs,
                               _CONFIDENCE_OF_KNOWN_TOKENS)
    # Gets mask lens for each sample in the batch according to the mask ratio.
    mask_len = jnp.expand_dims(
        jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
    # Keeps at least one of prediction in this round and also masks out at least
    # one and for the next iteration
    mask_len = jnp.maximum(
        1,
        jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

    # Adds noise for randomness
    rng, choice_rng = jax.random.split(rng)
    masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                  choice_temperature * (1. - ratio))
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  return final_state.final_seqs