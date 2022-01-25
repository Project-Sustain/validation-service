import os
import sys
import getopt
from . import worker_server


def print_usage():
    print("USAGE\n\tpython3 overlay [OPTIONS]\n")
    print("OPTIONS\n\t--master\t\tStarts the master server")
    print("\t--worker <master_hostname>\tStarts the worker server, connecting to the master specified\n")


def print_usage_and_exit():
    print_usage()
    exit(1)


def main():
    print(f"Running main({sys.argv})...")

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "mw:", ["master", "worker", "master_host="])

        master_hostname = None
        node_type = None

        for opt, arg in opts:
            if opt in ['-m', '--master']:
                node_type = "master"
            elif opt in ['--master_host']:
                master_hostname = arg
            elif opt in ['-w', '--worker']:
                node_type = "worker"

        if node_type == "master":
            print("Starting master node ... TODO")
        elif node_type == "worker":
            print(f"Starting worker server, with master {master_hostname}... TODO")
            worker_server.serve(master_hostname)

    except Exception as e:
        print(f"Error: {e}")
        print_usage_and_exit()


if __name__ == "__main__":
    main()
