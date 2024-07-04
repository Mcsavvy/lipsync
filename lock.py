import datetime
import os
import socket

import http.client
import sys


key_url = "https://gist.githubusercontent.com/Mcsavvy/af3e650ff694a3d60dbcd32c4dd7493d/raw/lipsync.txt"
keyfile = ".lock"
lock_date = datetime.datetime(2024, 7, 6)


def check_internet_connection():
    try:
        # Try to connect to a well-known website
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        return False


def get_key():
    if os.path.exists(keyfile):
        with open(keyfile, "r") as f:
            return f.read()
    conn = http.client.HTTPSConnection("gist.githubusercontent.com")
    conn.request(
        "GET",
        "/Mcsavvy/af3e650ff694a3d60dbcd32c4dd7493d/raw/lipsync.txt",
    )
    res = conn.getresponse()
    data = res.read()
    if res.status != 200:
        raise RuntimeError(f"Failed to get key: {data.decode('utf-8')}")
    return data.decode("utf-8")


def lock_app():
    if datetime.datetime.now() < lock_date:
        return
    if not check_internet_connection():
        print("Connect to the internet", file=sys.stderr)
        exit(1)
    key = get_key()
    if key != "ALLOW":
        print("App is locked", file=sys.stderr)
        exit(1)
    with open(keyfile, "w") as f:
        f.write(key)
    return