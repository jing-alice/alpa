import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
from jax.lib import xla_client
server_address = '10.233.124.194:22228'
print(server_address)
service_server = xla_client._xla.get_distributed_runtime_service(server_address, 2)
print('1111')
distributed_client = xla_client._xla.get_distributed_runtime_client(server_address, 0)
print('2222')
distributed_client.connect()
print('3333')
backend = xla_client.make_gpu_client(distributed_client, node_id=0, num_nodes=2)
print(backend.devices())
# import time
# time.sleep(30)
service_server.shutdown()