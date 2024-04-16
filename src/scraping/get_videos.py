import argparse

import json

import os

import yt_dlp

from yt_dlp.utils._utils import DownloadError

from tqdm import tqdm

def main(args):
    input = args.input
    
    with open(input, "r") as f:
        links = [x.strip() for x in f.readlines()]

    ydl_opts = {"paths": {"home": "/export/c10/ksasse/videos", "temp": "/export/c10/ksasse/videos"}}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for link in tqdm(links):
            try:
                ydl.download(link)

            except DownloadError:
                print(f"{link} is broken")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')

    args = parser.parse_args()

    main(args)