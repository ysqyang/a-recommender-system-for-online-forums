import sys, os
import logging

LOG_DIR        = os.path.join(os.path.dirname(sys.path[0]), 'logs')
RUN_LOG_NAME   = 'run'
SERVE_LOG_NAME = 'serve'
LOGGER_LEVEL   = logging.INFO
LEVELS         = {'info': logging.INFO}
LOG_FORMAT     = '[%(asctime)s] [%(name)-10s] [%(levelname)-8s] -- %(message)s'
MODE           = 'w'