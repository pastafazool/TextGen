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

textchars = sorted(list(set(sentence)))
int_to_char = dict((i, c) for i, c in enumerate(chars))
