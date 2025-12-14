# src/logger.py
import logging
import os
from datetime import datetime

# Creating log file
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
Logs_path=os.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(Logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(Logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s -%(levelname)s- [%(message)s]",
    level=logging.INFO,


)