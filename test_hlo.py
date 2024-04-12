import dataclasses
import os
from typing import Optional

import numpy as np
from alpa.util import xla_computation_to_mlir_text
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
# jax.config.update('jax_platform_name', 'cpu')
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
# from jax._src import xla_bridge as xb
from jax.lib import xla_bridge as xb
import jax.numpy as jnp
# from jax._src.compiler import get_compile_options
import jaxlib.xla_extension as xe
# from jax._src.interpreters import pxla
from numpy import dtype

backend = xb.get_backend('gpu')

input_shape = [(((), dtype('int32')), ((), dtype('int32'))), (((3, 3, 3, 32), dtype('float32')), ((3, 3, 3, 32), dtype('float32'))), (((3, 3, 32, 64), dtype('float32')), ((3, 3, 32, 64), dtype('float32'))), (((10,), dtype('float32')), ((10,), dtype('float32'))), (((64, 10), dtype('float32')), ((64, 10), dtype('float32'))), (((), dtype('int32')), ((), dtype('int32'))), (((3, 3, 3, 32), dtype('float32')), ((3, 3, 3, 32), dtype('float32'))), (((3, 3, 32, 64), dtype('float32')), ((3, 3, 32, 64), dtype('float32'))), (((10,), dtype('float32')), ((10,), dtype('float32'))), (((64, 10), dtype('float32')), ((64, 10), dtype('float32'))), (((3, 3, 3, 32), dtype('float32')), ((3, 3, 3, 32), dtype('float32'))), (((3, 3, 32, 64), dtype('float32')), ((3, 3, 32, 64), dtype('float32'))), (((10,), dtype('float32')), ((10,), dtype('float32'))), (((64, 10), dtype('float32')), ((64, 10), dtype('float32'))), (((50, 32, 32, 3), dtype('float32')), ((50, 32, 32, 3), dtype('float32'))), (((50,), dtype('int32')), ((50,), dtype('int32')))]

input = [[jnp.ones(ishape,dtype=itype) for ishape, itype in i_shape]  for i_shape in input_shape ]


def get_compile_options(num_replicas: int,
                        num_partitions: int,
                        device_assignment: np.ndarray,
                        use_spmd_partitioning: bool,
                        parameter_is_tupled_arguments: int,
                        build_random_seed: int,
                        spmd_propagation_to_outputs: bool = False):
    """Return CompileOptions for XLA compilation."""
    compile_options = xb.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=use_spmd_partitioning,
    )
    compile_options.parameter_is_tupled_arguments = (
        parameter_is_tupled_arguments)
    build_options = compile_options.executable_build_options
    build_options.seed = build_random_seed
    build_options.allow_spmd_sharding_propagation_to_output =\
        [spmd_propagation_to_outputs]
    # FIXME: re-enable the new runtime when everything is ready.
    debug_options = build_options.debug_options
    debug_options.xla_gpu_enable_xla_runtime_executable = False
    return compile_options

num_devices = 2
build_random_seed = 42
compile_options = get_compile_options(
    num_replicas=1,
    num_partitions=num_devices,
    device_assignment=np.arange(num_devices).reshape((1, -1)),
    use_spmd_partitioning=True,
    parameter_is_tupled_arguments=False,
    build_random_seed=build_random_seed,
    spmd_propagation_to_outputs=True)

rewrite_for_grad_acc = False
rewrite_grad_acc_indices = None
bypass_device_assignment_check = False
all_gather_threshold = 1152921504606846976
all_reduce_threshold = 1152921504606846976
enable_overlapping = False

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

with open("vgg_hlo0_compilation.txt", 'rb') as f:
    hlo = f.read()
    # print(hlo)

# hlo = xla_computation_to_mlir_text(xe.XlaComputation(hlo))
# print(hlo)
# print(compile_options)

with XlaPassContext({
        # Gradient accumulation rewrite:
        "auto_sharding::rewrite_for_grad_acc": rewrite_for_grad_acc,
        "auto_sharding::rewrite_indices": rewrite_grad_acc_indices,
        # Build options
        "build_option::bypass_device_assignment_check":
            bypass_device_assignment_check,

        # Communication combiner options
        "combiner::all_gather_threshold":
            all_gather_threshold,
        "combiner::all_reduce_threshold":
            all_reduce_threshold,
        "done-event::enable":
            enable_overlapping,
}):
    compiled = backend.compile(hlo, compile_options)
    out = compiled.execute_sharded_on_local_devices(
                input)
    print(out)

