import os
import jax
import jax.numpy as jnp

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurai.config import set_platform

set_platform(platform='gpu')

from flax.training import train_state
import neurai.nn as nn
from neurai.datasets.cifar import CIFAR10
from neurai.datasets.dataloader import DataLoader
from neurai.opt import sgd
from neurai.nn.layer.loss import softmax_cross_entropy
import numpy as nnp
from neurai.datasets.transforms import Compose, Normalize
import alpa
from alpa.pipeline_parallel.layer_construction import AutoLayerOption
from alpa.pipeline_parallel.stage_construction import UniformStageOption

class CNN(nn.Module):
  num_classes: int

  def __call__(self, x, **kwargs):
    x = nn.Conv(32, (3, 3))(x)
    x = x.reshape((x.shape[0], -1))
    x = nn.Linear(self.num_classes)(x)
    return x


@alpa.parallelize(method=alpa.PipeshardParallel(layer_option=AutoLayerOption(layer_num=2),stage_option=UniformStageOption(num_stages=2),num_micro_batches=4, \
  pipeline_schedule='gpipe'),batch_argnums=(1,2))
def train_step(state, data, label):
  def loss_fn(params):
    logits = state.apply_fn(params, data)
    one_hot = jax.nn.one_hot(label, 10)
    loss = nnp.mean(softmax_cross_entropy(logits, one_hot))
    return loss

  grad_fn = alpa.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss 

def create_train_state():
  """Creates initial `TrainState`."""
  cnn = CNN(10)
  params = cnn.init(jnp.ones([1, 32, 32, 3]), train=False)
  tx = sgd(0.01, 0.9)
  return train_state.TrainState.create(
      apply_fn=cnn.run, params=params, tx=tx)

DATASETS_DIR = '.'
# downLoad datasets
train_data = CIFAR10(
  DATASETS_DIR, train=True, download=True, transform=Compose([Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0))]))

# need add one_hot operation
train_loader = DataLoader(train_data, batch_size=100)

def train_and_test():
  alpa.init(cluster="ray")
  state = create_train_state()

  for epoch in range(1, 10000 + 1):
    with tqdm.tqdm(train_loader) as tepoch:
      tepoch.set_description(f"Training/epoch {epoch}")
      for batch in tepoch:
        state, loss = train_step(state, batch[0], batch[1])
        tepoch.set_postfix(loss=loss._value)

  return state

if __name__ == "__main__":
  train_and_test()


