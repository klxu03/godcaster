import argparse

import json

import os

import yt_dlp

from yt_dlp.utils._utils import DownloadError

from tqdm import tqdm

def main(args):
    input = args.input
    output = args.output
    
    process_id = args.id

    with open(input, "r") as f:
        links = [x.strip() for x in f.readlines()]

    infos = []

    with yt_dlp.YoutubeDL() as ydl:
        for link in tqdm(links):
            try:
                info = ydl.extract_info(link, download=False)

                infos.append(info)
                
            except DownloadError:
                infos.append(None)

    with open(os.path.join(output, f"info_{process_id}.json"), "w") as f:
        json.dump(infos, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--id')

    args = parser.parse_args()

    main(args)