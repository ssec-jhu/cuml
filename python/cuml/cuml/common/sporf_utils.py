#
# sporf_utils.py
#

import builtins
from datetime import datetime as dt


# emit timestamp and formatted string, separated by tab (ASCII 9)
def dtPrint(*args, **kwargs):
    builtins.print(f"{dt.now():%Y%m%d %H%M%S.%f}\t", end="")
    builtins.print("?", end="")
    builtins.print(*args, **kwargs)


# emit timestamp and formatted string, separated by space (ASCII 32)
def dtLog(*args, **kwargs):
    builtins.print(f"{dt.now():%Y%m%d %H%M%S.%f} ", end="")
    builtins.print(*args, **kwargs)


# format byte count as bytes and gigabytes
def sbgb(nBytes):
    return "{0} {1:.1F}GB".format(nBytes, nBytes / (1024.0 * 1024 * 1024))
