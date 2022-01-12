import os
import sys
import master


def print_usage():
    print("USAGE\n\tpython3 overlay [master|slave] [start|stop] [OPTIONS]")
    print("OPTIONS\n\t--master=<hostname>\tMust be specified when starting a slave")


def print_usage_and_exit():
    print_usage()
    exit(1)


def main():
    print(f"Running main({sys.argv})...")

    if 3 <= len(sys.argv) <= 4:
        node_type = sys.argv[1].lower()
        action = sys.argv[2].lower()

        if node_type == "slave":
            if action == "start" and len(sys.argv) == 4:
                print("TODO")
            elif action == "stop":
                print("TODO")
            else:
                print_usage_and_exit()
        elif node_type == "master":
            if action == "start":
                master.start_server()
            elif action == "stop":
                print("TODO")
            else:
                print_usage_and_exit()
    else:
        print_usage_and_exit()


if __name__ == "__main__":
    main()
