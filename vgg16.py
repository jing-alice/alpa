from copy import deepcopy
from functools import partial
import os
import jax
import jax.numpy as jnp

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from typing import Sequence, Tuple
from neurai.util.trans import jit
import tqdm
from neurai.grads import BP
from neurai.initializer import KaimingUniformIniter
import math

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurai.config import set_platform

set_platform(platform='gpu')

from flax.training import train_state
from neurai.nn.module import Module
import neurai.nn as nn
from neurai.nn.layer.pooling import MaxPool
from neurai.nn.layer.linear import Linear
from neurai.nn.layer.conv import Conv2d
from neurai.nn.layer.normalization import BatchNorm2d
from neurai.nn.layer.activate import Relu
from neurai.nn.layer.dropout import Dropout
from neurai.datasets.mnist import MNIST
from neurai.datasets.cifar import CIFAR10
from neurai.datasets.dataloader import DataLoader
from neurai.opt import sgd, adamw
from neurai.nn.layer.loss import softmax_cross_entropy
import numpy as nnp
from neurai.datasets.transforms import Compose, Normalize
import alpa
from alpa.pipeline_parallel.layer_construction import AutoLayerOption
from alpa.pipeline_parallel.stage_construction import UniformStageOption

print(os.getpid())
class VGG16(Module):
  act: Module = Relu
  conv: Module = Conv2d
  line: Module = Linear
  maxpool: Module = MaxPool

  def __call__(self, x, train: bool = False):
    conv = partial(
      self.conv, kernel_init=KaimingUniformIniter(scale=math.sqrt(5), distribution='leaky_relu'), use_bias=False)
    linear = partial(self.line, w_initializer=KaimingUniformIniter(scale=math.sqrt(5), distribution='leaky_relu'))

    x = conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)

    x = conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)
    x = MaxPool(window_shape=(2, 2), strides=(2, 2))(x)

    x = conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)

    x = conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)
    x = MaxPool(window_shape=(2, 2), strides=(2, 2))(x)

    x = conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)

    x = conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)

    x = conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)
    x = MaxPool(window_shape=(2, 2), strides=(2, 2))(x)

    x = conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)

    x = conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)

    x = conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)
    x = MaxPool(window_shape=(2, 2), strides=(2, 2))(x)

    x = conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)
    x = conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)
    x = conv(features=512, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
    x = self.act()(x)
    x = MaxPool(window_shape=(2, 2), strides=(2, 2))(x)

    x = nnp.mean(x, axis=(1, 2))

    x = linear(1024)(x)
    x = self.act()(x)

    x = linear(1024)(x)
    x = self.act()(x)

    x = linear(10)(x)


    return x
  
#@e(method=alpa.PipeshardParallel(stage_option=UniformStageOption(num_stages=2),num_micro_batches=10),batch_argnums=(1,2))
# @alpa.parallelize(batch_argnums=(1,2))
# @alpa.parallelize(method=alpa.PipeshardParallel(layer_option=AutoLayerOption(layer_num=2),stage_option=UniformStageOption(num_stages=2),num_micro_batches=10),batch_argnums=(1,2))
# @alpa.parallelize(method=alpa.ShardParallel(),batch_argnums=(1,2))
@alpa.parallelize(method=alpa.PipeshardParallel(layer_option=AutoLayerOption(layer_num=2), stage_option=UniformStageOption(num_stages=2),num_micro_batches=10),batch_argnums=(1,2))
def train_step(state, data, label):
  def loss_fn(params):
    logits = state.apply_fn(params, data)
    one_hot = jax.nn.one_hot(label, 10)
    loss = jnp.mean(softmax_cross_entropy(logits, one_hot))
    return loss, logits

  grad_fn = alpa.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss


#@alpa.parallelize(method=alpa.PipeshardParallel(stage_option=UniformStageOption(num_stages=2),num_micro_batches=10,pipeline_schedule="inference"),donate_argnums=(),batch_argnums=(1,2))
# @alpa.parallelize(batch_argnums=(1,2),donate_argnums=())
@alpa.parallelize(method=alpa.PipeshardParallel(stage_option=UniformStageOption(num_stages=2),num_micro_batches=10,pipeline_schedule="inference"),donate_argnums=(),batch_argnums=(1,2))
def test_step(state, data, label):
  logits = state.apply_fn(state.params, data)
  one_hot = jax.nn.one_hot(label, 10)
  loss = jnp.mean(softmax_cross_entropy(logits, one_hot))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == label)

  return loss, accuracy



DATASETS_DIR = '.'
# downLoad datasets
train_data = CIFAR10(
  DATASETS_DIR, train=True, download=True, transform=Compose([Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0))]))
test_data = CIFAR10(
  DATASETS_DIR, train=False, download=True, transform=Compose([Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0))]))

# need add ont_hot operation
train_loader = DataLoader(train_data, batch_size=100)
test_loader = DataLoader(test_data, batch_size=100)

def create_train_state():
  """Creates initial `TrainState`."""
  cnn = VGG16()
  params = cnn.init(jnp.ones([1, 28, 28, 3]), train=False)
  tx = adamw(learning_rate=0.0003, weight_decay=5e-4)
  return train_state.TrainState.create(
      apply_fn=cnn.run, params=params, tx=tx)

def train_and_test():
  alpa.init(cluster="ray")
  state = create_train_state()

  for epoch in range(1, 10000 + 1):
    with tqdm.tqdm(train_loader) as tepoch:
      tepoch.set_description(f"Training/epoch {epoch}")
      for batch in tepoch:
        state, loss = train_step(state, batch[0], batch[1])
        # jax.debug.visualize_array_sharding(state.params['param']['Conv2d_0']["weight"])
        tepoch.set_postfix(loss=loss._value)

    with tqdm.tqdm(test_loader) as tepoch:
      tepoch.set_description("Testing")
      for batch in tepoch:
        loss_val, acc_val = test_step(state, batch[0], batch[1])
        tepoch.set_postfix(loss=loss_val._value, acc=acc_val._value)

  return state

if __name__ == "__main__":
  train_and_test()
