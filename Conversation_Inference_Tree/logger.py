import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

handler = logging.FileHandler("log.txt")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)