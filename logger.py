import logging
import os
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the log file
log_file = os.path.join(base_dir, 'agent_log.txt')

# Setup logger
logging.basicConfig(
    filename=log_file,           # Log file name
    filemode='a',                       # Append to the file
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO                  # Log level: INFO, DEBUG, etc.
)
logger = logging.getLogger(__name__)
