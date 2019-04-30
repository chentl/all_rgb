import sys

# Use of pickle protocol 4 in cache.py requires Python 3.4 and higher

if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    raise Exception('Must be using at least Python 3.4')
else:
    from .filter import AllRGBFilter
    from .log import auto_log
