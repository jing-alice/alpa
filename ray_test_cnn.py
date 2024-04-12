import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tqdm

# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ray
import jax 
import alpa

# from neurai.config import set_platform

# set_platform(platform='gpu')


from flax.training import train_state
import neurai.nn as nn
from neurai.datasets.cifar import CIFAR10
from neurai.datasets.dataloader import DataLoader
from neurai.opt import sgd
from neurai.nn.layer.loss import softmax_cross_entropy
import numpy as nnp
from neurai.datasets.transforms import Compose, Normalize



from ray import tune, train
from ray.train import RunConfig, Checkpoint

from jax import numpy as jnp
from jax import random
from jax import grad, jit, vmap
from flax import linen as nn
from flax.training import train_state

storage_path = os.path.expanduser("~/ray_results")
exp_name = "tune_fault_tolerance_cnn"
path = os.path.join(storage_path, exp_name)

ray.init()
# jax.config.update("jax_platform_name", 'gpu')




# @jit
def forward(x, W, b):
  return jnp.dot(x, W) + b

# @jit
def loss(x, y, W, b):
  preds = forward(x, W, b)
  return jnp.mean(jnp.square(preds - y))

# @jit
def gradient(x, y, W, b):
  return grad(loss)(x, y, W, b)

x = jnp.array([[1.,2.,3.],[4.,5.,6.]], dtype=jnp.float32)  
y = jnp.array([2.,5.,10.], dtype=jnp.float32)

def trainable(config):
  print("config: ", config)
  epoches = config["epoch"]
  W = config["W"]
  b = config ["b"]
  
  lr = 0.00000001
  for i in range(epoches):
    grads = gradient(x, y, W, b)
    W -= lr * grads[0]  
    b -= lr * grads[1]

    print("Predicted:", forward(x, W, b))
    print("Loss:", loss(x, y, W, b))
  
param_space = {
  "epoch": 100000,
  "W" : jnp.array([[1., 2., 3.],[4.,5.,6.], [7., 8.,9.]], dtype=jnp.float32),
  "b" : jnp.array([0.5,1.,1.], dtype=jnp.float32) 
}

if tune.Tuner.can_restore(path):
    print("have falut, restore!")
    tuner = tune.Tuner.restore(path, trainable=trainable, resume_errored=True)
else:
    print("train")
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
    )

results = tuner.fit()
