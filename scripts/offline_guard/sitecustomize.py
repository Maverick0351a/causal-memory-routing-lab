"""Child-process guard for these demos; not an OS sandbox."""
import os
import sys

if os.environ.get("ARCHLAB_OFFLINE") == "1":
    def _deny(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
            raise RuntimeError("ARCHLAB_OFFLINE blocked " + event)
    sys.addaudithook(_deny)
