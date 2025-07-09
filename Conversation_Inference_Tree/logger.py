import logging
import time

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(f"/home/umflint.edu/brayclou/Conversation-Inference-Tree/log_{time.localtime}.txt")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)