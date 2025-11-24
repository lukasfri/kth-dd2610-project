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