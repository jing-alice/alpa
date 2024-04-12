import ray

from ray import tune, train
from ray.train import RunConfig, Checkpoint

from jax import numpy as jnp
from jax import random
# tuner = tune.Tuner(
#     trainable,
#     run_config=RunConfig(
#         name="experiment_name",
#         storage_path="s3://bucket-name/sub-path/",
#     )
# )
# tuner.fit()

import os


# Look for the existing cluster and connect to it
ray.init()

key = random.key(0)
# Set the local caching directory. Results will be stored here
# before they are synced to remote storage. This env variable is ignored
# if `storage_path` below is set to a local directory.

# os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = ""


storage_path = os.path.expanduser("~/ray_results")
exp_name = "tune_fault_tolerance_guide"
path = os.path.join(storage_path, exp_name)


def objective(x, a, b):  # Define an objective function.
    return a * (x**0.5) + b


def trainable(config):  # Pass a "config" dictionary into your trainable.
    epoch = config["epoch"]
    
    for x in range(epoch):  # "Train" for 20 iterations and compute intermediate scores.
        score = objective(x, config["a"], config["b"])

        train.report({"score": score})  # Send the score to Tune.




# param_space = {
#     "scaling_config": ScalingConfig(
#         num_workers=tune.grid_search([2, 4]),
#         resources_per_worker={
#             "CPU": tune.grid_search([1, 2]),
#         },
#     ),
#     # You can even grid search various datasets in Tune.
#     # "datasets": {
#     #     "train": tune.grid_search(
#     #         [ds1, ds2]
#     #     ),
#     # },
#     "params": {
#         "objective": "binary:logistic",
#         "tree_method": "approx",
#         "eval_metric": ["logloss", "error"],
#         "eta": tune.loguniform(1e-4, 1e-1),
#         "subsample": tune.uniform(0.5, 1.0),
#         "max_depth": tune.randint(1, 9),
#     },
# }

param_space = {
  "epoch": 2000000,
  "x": random.randint(key, shape=(1,), minval=5, maxval=20),
  "a": tune.uniform(0, 2),
  "b": tune.uniform(0, 1)  
}

# tuner = tune.Tuner(trainable=trainable, param_space=param_space,
#     run_config=RunConfig(name="my_tune_run"))


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

