"""HTTP server lifecycle; loopback by default, optional IPv4 interface binding."""

from ipaddress import IPv4Address
import threading


def serve_local(*, handler_class, server_class, port, background=(), output=print,
                host="127.0.0.1"):
    host = str(IPv4Address(host))
    server = server_class((host, int(port)), handler_class)
    try:
        for target in background:
            threading.Thread(target=target, daemon=True).start()
        output("Token Meter live -> http://{}:{}".format(host, port))
        if host == "0.0.0.0":
            output("All IPv4 interfaces; open localhost or this machine's LAN IP. "
                   "No authentication or TLS; use a trusted network only.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            output("\nstopped.")
    finally:
        server.server_close()
