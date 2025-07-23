import logging
import time

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)

handler = logging.FileHandler(f"./.logs/log_{current_time}.txt")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)

###################################################################################

log_progress = logging.getLogger(__name__)

log_progress.setLevel(logging.INFO)

progress_handler = logging.FileHandler(f"./.logs/progress_{current_time}.txt")

progress_handler.setFormatter(formatter)
log_progress.addHandler(progress_handler)
