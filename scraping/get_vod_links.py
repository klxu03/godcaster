import argparse

import os

import pickle

import re

import requests

from typing import List

from bs4 import BeautifulSoup

from selenium import webdriver

from selenium.webdriver.common.action_chains import ActionChains

from selenium.webdriver.common.by import By

from selenium.common.exceptions import NoSuchElementException, TimeoutException

from selenium.webdriver.support.wait import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

from tqdm import tqdm

def main(args):

    output = args.output

    r = requests.get("https://valorantesports.com/en-US/vods")

    soup = BeautifulSoup(r.text)

    links = [x.get("href") for x in soup.find_all("div", attrs={"role": "region"})[1].find_all("a")]

    total_vods = []

    for link in links:

        r = requests.get("https://valorantesports.com" + link)

        soup = BeautifulSoup(r.text)

        vods = [x.find("a").get("href") for x in soup.find_all("div", attrs={"data-spoiler": re.compile(".")})]

        total_vods.extend(vods)
    
    driver = webdriver.Chrome()

    driver.set_window_size(1920, 1080)
    vod_link = []

    broken_links = []

    for vod in tqdm(total_vods):
        try:
            driver.get("https://valorantesports.com" + vod)

            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//iframe[contains(@src, 'twitch') or contains(@src, 'youtube')]")))

            soup = BeautifulSoup(driver.page_source)

            vod_link.append(soup.find("iframe", attrs={"src": re.compile("twitch|youtube")}).get("src"))
        except TimeoutException:
            broken_links.append(vod)
    
    with open(os.path.join(output, "vod_links.pickle"), "wb") as f:
        pickle.dump(vod_link, f) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    main(args)