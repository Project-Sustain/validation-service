import sys
import getopt
import logging
from logging import info, error
import worker_server
import master_server
import flask_server


def print_usage():
    print("USAGE\n\tpython3 overlay [OPTIONS]\n")
    print("OPTIONS\n\t--master\t\tStarts the master server")
    print("\t--worker <master_hostname>\tStarts the worker server, connecting to the master specified\n")
    print("\t--flaskserver <master_hostname>\tStarts the flask server, connecting to the master specified\n")


def print_usage_and_exit():
    print_usage()
    exit(1)


def main():
    print(f"Running main({sys.argv})...")

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "mwfp:u:", ["master", "worker", "flaskserver", "port=", "master_uri="])

        node_type_arg = None
        port_arg = None
        master_uri_arg = None

        for opt, arg in opts:
            if opt in ['-m', '--master']:
                node_type_arg = "master"
            elif opt in ['-f', '--flaskserver']:
                node_type_arg = "flaskserver"
            elif opt in ['-w', '--worker']:
                node_type_arg = "worker"
            elif opt in ['--master_uri']:
                master_uri_arg = arg
            elif opt in ['-p', '--port']:
                port_arg = int(arg)

        if node_type_arg == "master":
            master_server.run(port_arg) if port_arg else master_server.run()
        elif node_type_arg == "worker":
            ok, master_hostname, master_port = is_valid_master_uri(master_uri_arg)
            if ok:
                if port_arg:
                    worker_server.run(master_hostname, master_port, port_arg)
                else:
                    worker_server.run(master_hostname, master_port)
        elif node_type_arg == "flaskserver":
            ok, master_hostname, master_port = is_valid_master_uri(master_uri_arg)
            if ok:
                if port_arg:
                    flask_server.run(master_hostname, master_port, port_arg)
                else:
                    flask_server.run(master_hostname, master_port)

    except Exception as e:
        print(f"Error: {e}")
        print_usage_and_exit()


def is_valid_master_uri(uri):
    if uri and uri != "":
        if ":" in uri:
            parts = uri.split(":")
            if len(parts) == 2:
                hostname = parts[0]
                try:
                    port = int(parts[1])
                    return True, hostname, port
                except ValueError as e:
                    error(f"Unable to parse port number: {e}")
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
