import numpy as np
from typing import Optional, Sequence 

class DistributedArray:
    """A distributed array on a PhysicalDeviceMesh.

    End users can interact with this array as if they are working with
    a normal numpy array.

    Internally, it stores a pointer to all remote buffers.
    The buffers are stored distributedly on remote workers' device memory.
    When users require the value of the array. These buffers will be gathered
    to the driver.
    """

    def __init__(self,
                 device_mesh: PhysicalDeviceMesh,
                 aval: ShapedArray,
                 sharding_spec: ShardingSpec,
                 remote_ref: RemoteArrayRef,
                 indices: Optional[Sequence[Index]] = None):
        self.device_mesh = device_mesh
        self.aval = aval
        self.sharding_spec = sharding_spec
        self.remote_ref = remote_ref

        if indices is None:
            indices = spec_to_indices(self.aval.shape, self.sharding_spec)
        self.indices = indices

        self.shape = self.aval.shape
        self.dtype = self.aval.dtype
        self._npy_value = None
        self._one_replica_host_local_ids = None
        self._one_replica_buffer_ids = None
        self._fetched_np_buffers = None
        self._fetched_np_buffers_ref = None
        self.skip_shard_args_check = False

    @property
    def size(self):
        return np.prod(self.shape)

    def _compute_one_replica_ids(self):
        one_replica_indices = []
        one_replica_host_local_ids = []
        seen_index_hashes = set()
        for i, index in enumerate(self.indices):
            hashed_index = _hashable_index(index)
            if hashed_index not in seen_index_hashes:
                one_replica_indices.append(i)
                one_replica_host_local_ids.append(
                    divmod(i, self.device_mesh.num_devices_per_host))
                seen_index_hashes.add(hashed_index)
        self._one_replica_buffer_ids = one_replica_indices
        self._one_replica_host_local_ids = one_replica_host_local_ids

    # TODO(yonghao): to make ._value faster(in reorder buffer), cache different
    # buffers with the same mesh shape and sharding spec.
    @property
    def one_replica_buffer_ids(self):
        """Indices of buffers containing one complete copy of the array data."""
        if self._one_replica_buffer_ids is None:
            self._compute_one_replica_ids()
        return self._one_replica_buffer_ids

    @property
    def one_replica_host_local_ids(self):
        if self._one_replica_host_local_ids is None:
            self._compute_one_replica_ids()
        return self._one_replica_host_local_ids

    @property
    def _value(self):
        if self._npy_value is None:
            print("remote_ref: ", self.remote_ref)
            print("one_replica_host_local_ids: ", self.one_replica_host_local_ids)
            npy_value = np.empty(self.aval.shape, self.aval.dtype)
            if not self._fetched_np_buffers:
                if not self._fetched_np_buffers_ref:
                    fetched_np_buffers = self.device_mesh.get_remote_buffers(
                        (self.remote_ref,),
                        (self.one_replica_host_local_ids,))[0]
                else:
                    fetched_np_buffers = ray.get(self._fetched_np_buffers_ref)
            else:
                fetched_np_buffers = self._fetched_np_buffers 
            for ct, i in enumerate(self.one_replica_buffer_ids):
                npy_value[self.indices[i]] = fetched_np_buffers[ct]
            self._npy_value = npy_value
        return self._npy_value

    def __array__(self, dtype=None, context=None):
        # pylint: disable=unused-argument
        return np.asarray(self._value, dtype=dtype)

    def __float__(self):
        return self._value.__float__()

    def __str__(self):
        return (f"DistributedArray(sharding_spec={self.sharding_spec}, "
                f"value={self._value})")

    def __del__(self):
        self.delete()


# core.pytype_aval_mappings[DistributedArray] = attrgetter("aval")
# xla.pytype_aval_mappings[DistributedArray] = attrgetter("aval")
# xla.canonicalize_dtype_handlers[DistributedArray] = lambda x: x



