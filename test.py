import numpy as np

with open('H:/Workspaces/Walk-Assistant/data/frames/annotation.txt', 'r') as f:
    lines = f.readlines()

text = ''
for index, line in enumerate(lines):
    if "GOPR5068.MP4_2772.jpg" in line:
        text = line

print(text)

file, encode = str(text).replace('\n', '').split(',')

arr = []
for b in encode:
    arr.append(int(b))

arr = np.array(arr).reshape((9, 16))

print(arr)
