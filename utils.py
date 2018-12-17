import os
import csv
import cv2
import numpy as np
import random
import sklearn
import tqdm
IMAGE_STAT_DIR = "image_stats"
IMAGE_MEAN_PATH = os.path.join(IMAGE_STAT_DIR, "image_mean.npy")
IMAGE_STD_PATH = os.path.join(IMAGE_STAT_DIR, "image_std.npy")
DATA_PATH = "data"


def read_csv_file(path):
    samples = []
    assert os.path.isfile(path), "Path is not a file: {}".format(path)
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples[1:]

def preprocess_img(im):
    im = (im / 255).astype(np.float32)
    im = im[:, :, ::-1]
    return im

def read_img(path):
    im = cv2.imread(path)
    im = preprocess_img(im)
    return im




def load_image_stats():
    if os.path.isfile(IMAGE_STD_PATH) and os.path.isfile(IMAGE_MEAN_PATH):
        return np.load(IMAGE_MEAN_PATH), np.load(IMAGE_STD_PATH)
    print("Did not find image mean and std file. Creating")
    os.makedirs(IMAGE_STAT_DIR, exist_ok=True)
    samples = read_csv_file()
    images = np.zeros((len(samples)*3, 160, 320, 3), dtype=np.float16)
    for i, batch_sample in enumerate(tqdm.tqdm(samples, desc="Generating image mean/std")):
        for idx in [0, 1, 2]:
            j = i*3 + idx
            name = batch_sample[idx].split("/")[-1]
            filepath = os.path.join(DATA_PATH, "IMG", name)
            img = read_img(filepath)
            images[j] = img
    print("Computing mean and std from shape:", images.shape)
    print("This will take some time..")
    mean = images.mean(axis=(0,-1)).astype(np.float32)[:, :, None]
    std = images.std(axis=(0,-1)).astype(np.float32)[:, :, None]
    np.save(IMAGE_MEAN_PATH, mean)
    np.save(IMAGE_STD_PATH, std)
    return mean, std



if __name__ == "__main__":
    load_image_stats()