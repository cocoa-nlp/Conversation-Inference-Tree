import logging
import time
import os

# Time format for filenames
t = time.localtime()
current_time = time.strftime("%H-%M-%S", t)  # Avoid colon ":" in filenames (especially on Windows)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, '.logs')
os.makedirs(LOG_DIR, exist_ok=True)

# File paths (use f-strings for actual variable substitution)
LOG_FILE = os.path.join(LOG_DIR, f'log_{current_time}.log')
PROGRESS_FILE = os.path.join(LOG_DIR, f'progress_{current_time}.log')

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
