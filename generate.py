import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import sys

import os, glob
import nltk

from sklearn import preprocessing
import test

import test

def main():
    print("Shakespeare TextGen");
    hello = test.TextFile("hello.txt", 5, 64, 0);
    hello.generate_text("models/model.hdf5");

if __name__ == '__main__':
    main()
