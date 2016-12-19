import pickle
import numpy as np
import math

from zipfile import ZipFile
from urllib.request import urlretrieve
import os


class Data():
    def __init__(self, download_path):
        self.download(download_path, "data.zip")
        self.uncompress("data.zip")
        train_path = "./Dataset/train.p"
        test_path = "./Dataset/test.p"

        with open(train_path, mode='rb') as f:
            train = pickle.load(f)
        with open(test_path, mode='rb') as f:
            test = pickle.load(f)

        # self.train_features, self.train_labels = self.randomize_data(train["features"], train["labels"])
        # print(self.train_labels.__class__)
        self.train_features = train['features']
        self.train_labels = train['labels']
        self.test_features = test['features']
        self.test_labels = test['labels']

    def download(self, url, file):
        """
        Download file from <url>
        :param url: URL to file
        :param file: Local file path
        """
        if not os.path.isfile(file):
            print('Downloading ' + file + '...')
            urlretrieve(url, file)
            print('Download Finished')

    def uncompress(self, file):
        if not os.path.isdir("Dataset"):
            print("Uncompressing Data..")
            with ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall("Dataset")
            print("Data is ready to use")

    def train_data(self):
        return self.train_features, self.train_labels


    def test_data(self):
        return self.test_features, self.test_labels










