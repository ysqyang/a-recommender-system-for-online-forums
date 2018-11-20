import sys, os
import logging

_LOG_DIR        = os.path.join(os.path.dirname(sys.path[0]), 'logs')
_RUN_LOG_NAME   = 'run'
_SERVE_LOG_NAME = 'serve'
_LOGGER_LEVEL   = logging.INFO
_LEVELS         = {'info': logging.INFO}
_LOG_FORMAT     = '[%(asctime)s] [%(name)-10s] [%(levelname)-8s] -- %(message)s'
_MODE           = 'a'