#!/bin/bash

wallet_name="$1"  # First command line argument
start_hotkey="$2" # Second command line argument
end_hotkey="$3"   # Third command line argument

if [ -z "$wallet_name" ] || [ -z "$start_hotkey" ] || [ -z "$end_hotkey" ]; then
    echo "Usage: $0 <wallet_name> <start_hotkey> <end_hotkey>"
    exit 1
fi

# Determine the IPs and base port based on the wallet name
case "$wallet_name" in
    1)
        external_ip="15.156.239.177"
        internal_ip="172.31.44.33"
        base_port=12500
        ;;
    2)
        external_ip="99.79.172.14"
        internal_ip="172.31.37.54"
        base_port=20000
        ;;
    3)
        external_ip="35.182.122.531"
        internal_ip="172.31.35.62"
        base_port=22000
        ;;
    4)
        external_ip="15.223.134.146"
        internal_ip="172.31.46.82"
        base_port=45000
        ;;
    5)
        external_ip="35.173.55.145"
        internal_ip="172.31.41.255"
        base_port=30000
        ;;
    *)
        echo "Error: Invalid wallet name"
        exit 1
        ;;
esac

chain_endpoint=194.163.169.70:9944

# Loop over the instances
for instance in $(seq $start_hotkey $end_hotkey); do
    port=$((base_port + instance))
    name="${wallet_name}:${instance}"

    # Check if the port exceeds 55000, if so, exit
    if [ "$port" -gt 55000 ]; then
        echo "Error: Port exceeds 55000. Stopping."
        exit 1
    fi

    echo "Starting instance $name on port $port"

    pm2 start endpooint_miner2.py --name $name --time --interpreter python3 -- --axon.port $port --wallet.name $wallet_name --netuid 1 --wallet.hotkey $instance --subtensor.chain_endpoint $chain_endpoint --logging.debug --axon.external_ip $external_ip --axon.external_port $port --axon.ip $internal_ip
done