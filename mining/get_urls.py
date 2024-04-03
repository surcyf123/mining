def generate_endpoints(input_text, with_generate=True):
    lines = input_text.strip().split('\n')
    endpoints = []

    for line in lines:
        ip, port_mapping = line.split(':')
        external_port = port_mapping.split(' ')[0]
        if with_generate:
            endpoint = f"'http://{ip.strip()}:{external_port}/generate',"
        else:
            endpoint = f"'http://{ip.strip()}:{external_port}',"
        endpoints.append(endpoint)
    
    return endpoints

# IP and port mappings directly in the script
mappings = """
211.21.106.84:34436 -> 30000/tcp
211.21.106.84:34432 -> 30001/tcp
211.21.106.84:34491 -> 30002/tcp
211.21.106.84:34400 -> 30003/tcp
"""

with_generate = False  # Change to False if you don't want /generate
endpoints = generate_endpoints(mappings, with_generate)

for endpoint in endpoints:
    print(endpoint)
