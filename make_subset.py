from data_loader import DataLoader
import random
import os
from shutil import copy2
from tqdm import tqdm

SRC = "H:/Workspaces/Walk-Assistant/data/frames"
DEST = "data/subset"
SUBSET = 100

os.makedirs(DEST, exist_ok=True)

files = DataLoader.get_files_list('{}/*.jpg'.format(SRC))
random.shuffle(files)

for file in tqdm(files[:SUBSET]):
    copy2(file, DEST)
