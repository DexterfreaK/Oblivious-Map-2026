"""ORAM/OMAP Server Demo.

Usage:
    python server.py [--ip IP] [--port PORT]

Examples:
    python server.py
    python server.py --port 6666
"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from daoram.dependency import RemoteServer, ZMQSocket


def main():
    parser = argparse.ArgumentParser(description="ORAM/OMAP Server")
    parser.add_argument("--ip", default="*", help="IP to bind (default: *)")
    parser.add_argument("--port", type=int, default=5555, help="Port (default: 5555)")
    args = parser.parse_args()

    print(f"Server listening on {args.ip}:{args.port}...")
    socket = ZMQSocket(ip=args.ip, port=args.port, is_server=True)
    server = RemoteServer()

    try:
        server.run(server=socket)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
