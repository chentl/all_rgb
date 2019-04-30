import sys


if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    raise Exception('Must be using at least Python 3.4')
else:
    from .filter import AllRGBFilter, FilterError
    from .log import auto_log
