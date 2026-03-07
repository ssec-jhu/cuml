#
# scaffold.py
#

import builtins
from datetime import datetime as dt


# emit timestamp and formatted string, separated by tab (ASCII 9)
def dtPrint(*args, **kwargs):
    builtins.print(f"{dt.now():%Y%m%d %H%M%S.%f}", end="\t")
    builtins.print(*args, **kwargs)


# emit timestamp and formatted string, separated by space (ASCII 32)
def dtLog(*args, **kwargs):
    builtins.print(f"{dt.now():%Y%m%d %H%M%S.%f}", end=" ")
    builtins.print(*args, **kwargs)


# emit byte count formatted as bytes and gigabytes
def sbgb(nBytes):
    return "{0} {1:.1F}GB".format(nBytes, nBytes / (1024.0 * 1024 * 1024))
