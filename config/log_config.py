import sys, os, logging

_LOG_DIR        = os.path.join(os.path.dirname(sys.path[0]), 'logs')
_NAME           = 'run'
_LOGGER_LEVEL   = logging.DEBUG
_LEVELS         = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
_LOG_FORMAT     = '[%(asctime)s] [%(name)s] [%(levelname)s] -- %(message)s'
_MODE           = 'a'