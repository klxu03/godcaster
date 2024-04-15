import numpy as np

import pickle

from urllib.parse import unquote

with open("vod_links_total", "rb") as f:
    urls = pickle.load(f)


for i in range(len(urls)):
    if "youtube" in urls[i]:
        urls[i] = urls[i].split("?")[0]
    else:
         urls[i] = "https://www.twitch.tv/videos/" + unquote(urls[i].split("&")[-2].replace("video=", ""))

urls = list(set(urls))

splits = np.array_split(urls, 10)

for i,split in enumerate(splits):
    with open(f"data/splits/split_{i}.txt", "w") as f:
        f.write("\n".join(split))