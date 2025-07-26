import logging
import time
import os
from appdirs import user_log_dir

# Time format for filenames
t = time.localtime()
current_time = time.strftime("%H-%M-%S", t)

# Directory setup
BASE_DIR = user_log_dir("ConversationInferenceTree")
os.makedirs(BASE_DIR, exist_ok=True)

# File paths (use f-strings for actual variable substitution)
LOG_FILE = os.path.join(BASE_DIR, f'log_{current_time}.log')
PROGRESS_FILE = os.path.join(BASE_DIR, f'progress_{current_time}.log')

# Logger for general debug/info
logger = logging.getLogger('main_logger')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Logger for progress (separate file)
log_progress = logging.getLogger('progress_logger')
log_progress.setLevel(logging.INFO)

progress_handler = logging.FileHandler(PROGRESS_FILE)
progress_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
log_progress.addHandler(progress_handler)
