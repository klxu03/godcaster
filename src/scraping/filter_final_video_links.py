import json
import os
import numpy as np
info_dir = "/home/ksasse1/code/godcaster/data/info"

files = [os.path.join(info_dir, x) for x in os.listdir(info_dir)]
total = []
for file in files:
    with open(file, "r") as f:
        total += json.load(f)
total = [x for x in total if x is not None]
acceptable_channels = {'VALORANT Challengers Americas',
'VALORANT Champions Tour',
'VALORANT Champions Tour Americas',
'VALORANT Champions Tour EMEA',
'VALORANT'}
titles_filtered = []

for i,title in enumerate(total):
    if title["uploader"] in acceptable_channels:
        titles_filtered.append(title)
final_urls = [x['webpage_url'] for x in titles_filtered]
splits = np.array_split(final_urls, 35)

for i,split in enumerate(splits):
    with open(f"/home/ksasse1/code/godcaster/data/video_splits/split_{i}.txt", "w") as f:
        f.write("\n".join(split))
