import sys, os
import logging

_LOG_DIR        = os.path.join(os.path.dirname(sys.path[0]), 'logs')
_NAME           = 'run'
_LOGGER_LEVEL   = logging.INFO
_LEVELS         = {'INFO': logging.INFO, 
                   'ERROR': logging.ERROR
                  }
_LOG_FORMAT     = '[%(asctime)s] [%(name)-10s] [%(levelname)-8s] -- %(message)s'
_MODE           = 'a'