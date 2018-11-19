import sys, os

_LOG_DIR        = os.path.join(os.path.dirname(sys.path[0]), 'logs')
_NAME           = 'run'
_LOGGER_LEVEL   = 'INFO'
_LEVELS         = {'DEBUG':    10,
                   'INFO':     20,
                   'WARNING':  30,
                   'ERROR':    40,
                   'CRITICAL': 50
                   }
_LOG_FORMAT     = '[%(asctime)s] [%(name)-10s] [%(levelname)-8s] -- %(message)s'
_MODE           = 'a'

print(_LOG_DIR)