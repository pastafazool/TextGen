import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

import os, glob
import nltk

from sklearn import preprocessing

import test


def main():
    print("Shakespeare TextGen");
    hello = test.TextFile("hello.txt", 15, 128, 0.6);
    hello.run_lstm()

if __name__ == '__main__':
    main()
