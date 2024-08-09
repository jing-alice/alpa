import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import alpa
from tqdm import tqdm
from alpa.pipeline_parallel.stage_construction import UniformStageOption
from alpa.pipeline_parallel.layer_construction import AutoLayerOption
from neurai.datasets.cifar import CIFAR10
from neurai.datasets.dataloader import DataLoader
from neurai.datasets.transforms import Compose, Normalize
import tqdm
import os
pid = os.getpid()
print("当前进程的PID为：",pid)




class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    # x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    # x = nn.relu(x)
    # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    # x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    # x = nn.relu(x)
    # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


@alpa.parallelize(donate_argnums=(),method=alpa.PipeshardParallel(stage_option=UniformStageOption(num_stages=2),num_micro_batches=4))
def train_step(state, batchs):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batchs[0])
    one_hot = jax.nn.one_hot(batchs[1], 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss

  # grads = alpa.grad(loss_fn)(state.params) 
  grad_fun = alpa.value_and_grad(loss_fn)
  # grad_fun = jax.value_and_grad(loss_fn)
  loss, grads = grad_fun(state.params)
  # accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  state = state.apply_gradients(grads=grads)
  
  return state, loss, grads

# @alpa.parallelize(method=alpa.ShardParallel(),donate_argnums=())
# @alpa.parallelize(method=alpa.PipeshardParallel(layer_option=AutoLayerOption(layer_num=1),stage_option=UniformStageOption(num_stages=1),pipeline_schedule="inference"),donate_argnums=())
@alpa.parallelize(method=alpa.PipeshardParallel(stage_option=UniformStageOption(),pipeline_schedule="inference"),donate_argnums=())
def test_step(state, data, label):
  logits = state.apply_fn({'params': state.params}, data)
  one_hot = jax.nn.one_hot(label, 10)
  loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == label)
  return loss, accuracy

def create_train_state(rng):
  """Creates initial `TrainState`."""
  cnn = CNN()
  params = cnn.init(rng, jnp.ones([1, 32, 32, 3]))['params']
  tx = optax.sgd(0.01)
  return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


alpa.init(cluster="ray")

DATASETS_DIR = '.'
# downLoad datasets
train_data = CIFAR10(
  DATASETS_DIR, train=True, download=True, transform=Compose([Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0))]))
test_data = CIFAR10(
  DATASETS_DIR, train=False, download=True, transform=Compose([Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0))]))

# need add ont_hot operation
train_loader = DataLoader(train_data, batch_size=100)
test_loader = DataLoader(test_data, batch_size=100)

rng = jax.random.PRNGKey(0)
state = create_train_state(rng)
for epoch in range(1, 10 + 1):
    with tqdm.tqdm(train_loader) as tepoch:
      tepoch.set_description(f"Training/epoch {epoch}")
      for batch in tepoch:
        state, loss, grads = train_step(state, batch)
        # print("data: ", batch[0])
        tepoch.set_postfix(loss=loss._value)
      print("grads: ", grads['Dense_0']['kernel'])

    with tqdm.tqdm(test_loader) as tepoch:
      tepoch.set_description("Testing")
      for batch in tepoch:
        loss_val, acc_val = test_step(state, batch[0], batch[1])
        tepoch.set_postfix(loss=loss_val._value, acc=acc_val._value)





# for _ in range(10):
#   for _ in tqdm(range(100)): 
#     state, loss, grad = train_step(state, (image, label))
#   print("loss: ", loss._value)
#   jax.debug.print("grads: {}",grad['Conv_0']['kernel'])
#     # print("acc: ", acc)

# loss, acc = test_step(state, image[0:1], label[0])
# print("loss: ", loss._value)
# print("acc: ",acc._value)