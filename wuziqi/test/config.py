import logging

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format=LOG_FORMAT)