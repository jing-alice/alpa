import functools
import itertools as it
import logging
import os
import subprocess
import re
import socket
import time
from collections import OrderedDict
from functools import partial, partialmethod
import threading
from typing import Iterable, Dict, Sequence, Any, List
from warnings import warn

from flax.training import train_state
from flax.training.common_utils import stack_forest
import jax
from jax._src.util import wrap_name
from jax._src.source_info_util import SourceInfo
import jax.numpy as jnp
from jax._src import dispatch, source_info_util
from jax._src.api import ShapeDtypeStruct
from jax.lib import (
    xla_bridge as xb,
    xla_client as xc,
    xla_extension as xe
)
from jax.api_util import shaped_abstractify
from jax import core
from jax.core import (Atom, ClosedJaxpr, DropVar, Jaxpr, JaxprEqn, Literal,
                      Primitive, ShapedArray, Var, AbstractValue, gensym)
from jax._src.maps import FrozenDict
from jax import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla, pxla, mlir
# from jax.interpreters.xla import make_device_array
from jax._src.array import make_array_from_callback
from jax.tree_util import tree_map, tree_flatten, PyTreeDef
import numpy as np
import ray
from ray.util.placement_group import get_current_placement_group,\
    PlacementGroup
import tqdm

from alpa import device_mesh
from alpa.global_env import global_config, is_worker
from alpa.monkey_patch import (restore_random, monkey_patch_random,
                               rng_primitives)
from alpa.wrapped_hlo import HloStatus, WrappedHlo
from jax.lib import xla_bridge as xb
# from jax._src.compiler import get_compile_options
import jaxlib.xla_extension as xe
backend = xb.get_backend('gpu')

class XlaPassContext:
    """A global context for passing arguments from python to XLA c++ passes."""

    current = None

    def __init__(self, value_dict):
        self.value_dict = value_dict

    def __enter__(self):
        assert XlaPassContext.current is None, ("Do not support nested context")
        XlaPassContext.current = self
        xe.set_pass_context(self.value_dict)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        XlaPassContext.current = None
        xe.clear_pass_context()
        
def xla_computation_to_mlir_text(xla_computation: xc.XlaComputation):
    return xc._xla.mlir.xla_computation_to_mlir_module(xla_computation)
  
def compile_allocate_zero_buffers(backend, num_devices: int,
                                  shapes: Sequence[Sequence[int]],
                                  dtypes: Sequence[jnp.dtype]):
    """Compile an XLA executable that returns zero buffers with given shape and
    dtypes."""
    c = xc.XlaBuilder("allocate_zero_buffers")
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    c.set_sharding(sharding)
    ret = []
    for shape, dtype in zip(shapes, dtypes):
        if dtype == "V2":
            dtype = jnp.bfloat16

        zero = xc.ops.Constant(c, jnp.array(0, dtype=dtype))
        zero = xc.ops.Broadcast(zero, shape)
        ret.append(zero)
    c.clear_sharding()
    c = c.build(xc.ops.Tuple(c, ret))

    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    build_options = compile_options.executable_build_options
    build_options.allow_spmd_sharding_propagation_to_output = [True]
    with XlaPassContext({
            "done-event::enable": global_config.enable_overlapping,
    }):
        compiled = backend.compile(xla_computation_to_mlir_text(c),
                                   compile_options)
    return compiled
  
  
buffers = compile_allocate_zero_buffers(backend, 1, [(4, 640, 11008), (4, 640, 4096), (4, 640, 4096)], [jnp.dtype('float32'), jnp.dtype('float32'), jnp.dtype('float32')])