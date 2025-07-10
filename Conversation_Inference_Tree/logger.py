import logging
import time

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)

handler = logging.FileHandler(f"/home/umflint.edu/brayclou/Conversation-Inference-Tree/.logs/log_{current_time}.txt")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)