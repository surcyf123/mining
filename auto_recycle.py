import pexpect
import re
import sys
from datetime import datetime

wallet = "2"
hotkeys = [str(i) for i in range(1, 36)]
highest_cost = 2.0
password = ""
MAX_RETRIES = 3  # Number of retries before skipping to the next hotkey


def get_clean_cost(child):
    cost_str = child.match.group(1).decode('utf-8').replace('τ', '')
    return float(re.sub(r'\x1b\[[0-9;]*m', '', cost_str).strip())


def register_hotkey(wallet, hotkey):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            command = 'btcli recycle_register -subtensor.network finney --netuid 1 --wallet.name {} --wallet.hotkey {}'.format(wallet, hotkey)
            print("\nColdkey:", wallet, "Hotkey:", hotkey, "Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
            child = pexpect.spawn(command)
            child.logfile_read = sys.stdout.buffer
            child.expect('Enter subtensor network')
            child.sendline('')

            child.expect(r'The cost to register by recycle is (.*?)(?:\n|$)')
            cost = get_clean_cost(child)
            if cost > highest_cost:
                child.sendline('n')
                retries += 1
                continue

            child.sendline('y')
            child.expect('Enter password to unlock key')
            child.sendline(password)

            child.expect(r'Recycle (.*?) to register on subnet')
            recycle_cost = get_clean_cost(child)
            if recycle_cost > highest_cost:
                child.sendline('n')
                retries += 1
                continue

            child.sendline('y')
            child.expect(r'Registered', timeout=120)
            return
        except Exception as e:
            print("An error occurred", e)
            retries += 1

    print(f"Failed to register hotkey {hotkey} after {MAX_RETRIES} attempts.")


def main():
    for hotkey in hotkeys:
        register_hotkey(wallet, hotkey)


if __name__ == "__main__":
    main()
