"""
Walk-Assistant : Recognizing sidewalk for the visually impaired
Copyright (C) 2018 Yoongi Kim (devlifecode@outlook.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
print("""
    Walk-Assistant Copyright (C) 2018 Yoongi Kim
    This program comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it
    under certain conditions.
""")


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
