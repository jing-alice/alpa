filename = '/etc/mpiworker.ip'
output_filename = 'hostfile'

import socket
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

with open(filename, 'r') as file:
    content = file.read().strip()

hosts = content.split(',')
rs = []
for host in hosts[1:]:
    ip = host.split(':')[0]
    if ip == ip_address:
        rs.insert(0, ip)
    else:
        rs.append(ip)
print(rs)

with open(output_filename, 'w') as output_file:
    for host in rs:
        output_file.write(f'{host}\tslots=1\n')