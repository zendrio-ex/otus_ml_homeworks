import os
from src.config import settings

# It can be done via import gdown
command = f"gdown --folder '{settings['google_drive']}'"
os.system(command)
