from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import alpa
from tqdm import tqdm
from alpa.pipeline_parallel.stage_construction import UniformStageOption


@alpa.parallelize(method=alpa.PipeshardParallel(stage_option=UniformStageOption(num_stages=2)))
def train_step(state, batchs):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits = state.apply_fn(params, batchs[0])
    one_hot = jax.nn.one_hot(batchs[1], 64)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss

  grads = alpa.grad(loss_fn)(state.params)
  state = state.apply_gradients(grads=grads)
  return state


def create_train_state(rng):
  """Creates initial `TrainState`."""
  def cnn(weight, x):

    x =  x @ weight
    x = nn.relu(x)
  
    alpa.mark_pipeline_boundary()

    x =  x @ weight
    x = nn.relu(x)

    return x

  params = jnp.ones([64,64])

  tx = optax.sgd(0.1)

  return train_state.TrainState.create(apply_fn=cnn, params=params, tx=tx)


alpa.init(cluster="ray")
image = jnp.ones([16, 64])
label = jnp.array([7, 8, 6, 5, 0, 4, 2, 1, 7, 8, 6, 5, 0, 4, 2, 1])
rng = jax.random.PRNGKey(0)
state = create_train_state(rng)
for _ in range(10):
  for _ in tqdm(range(10000)): 
    state=train_step(state, (image, label))
