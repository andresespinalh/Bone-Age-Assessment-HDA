import numpy as np
import cv2
import os
import pandas as pd

# Load images ids
def get_ids(folder):
    filenames = os.listdir(folder)
    ids = [os.path.splitext(filename)[0] for filename in filenames]
    return ids

# Load images and names
def load_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(os.path.splitext(filename)[0])
    return np.array(images), filenames

# Load the labels of the images
def get_labels(labels_path, filenames_list, key_col, value_col):
    df = pd.read_csv(labels_path)
    labels = []
    for value in filenames_list:
        row = df.loc[df[key_col] == int(value)]
        labels.append(row[value_col].values[0])
    return np.array(labels)