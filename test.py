import tensorflow as tf
import numpy as np
import keras

import os, glob
import nltk

from sklearn import preprocessing

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = preprocessing.LabelEncoder()


class TextFile(object):
    def __init__(self, filename, traintextlength):
        self.filename = filename
        self.traintextlength = traintextlength

    def tokenize(self):
        with open(self.filename, 'r') as sentencedata:
            sentence = sentencedata.read()
            print(sentence)
        sentence = str(sentence).lower()
        tokenize = nltk.word_tokenize(sentence)
        textpos = nltk.pos_tag(tokenize)

        textchars = sorted(list(set(sentence)))
        textcharinteger = dict((c, i) for i, c in enumerate(textchars))
        print(textchars)

        empty_wordlist = []

        for i in range(len(tokenize)):
            tokenize_result = tokenize[i]
            empty_wordlist.append(tokenize_result)

        wordarray = np.array(empty_wordlist)
        #print(wordarray)

        return sentence, textcharinteger, textcharinteger

    def onehotencode(self):
        origtext, textchars, integerversion = self.tokenize()
        print("Total Alphabet Size: " + str(len(textchars)))
        lstm_input = []
        lstm_output = []
        for character in range(0, len(origtext) - self.traintextlength):
            input = origtext[character : character + self.traintextlength]
            output = origtext[character + self.traintextlength]
            lstm_input.append([integerversion[characterval] for characterval in input])
            lstm_output.append(integerversion[output])
        training_data_size = len(lstm_input)
        output_data_size = len(lstm_output)

        print(training_data_size)
        print(output_data_size)

        return lstm_input, lstm_output, training_data_size

    def run_lstm(self):
        X, Y, input_datasize = self.onehotencode()
        #X = np.reshape(X, (input_datasize, self.traintextlength, 1))
        X = np.reshape(X, (input_datasize, self.traintextlength))
        X = X / float(input_datasize)
        print(X)
        print("---")

def main():
    print("Shakespeare TextGen");
    hello = TextFile("hello.txt", 15);
    #hello.tokenize()
    hello.onehotencode()
    hello.run_lstm()

if __name__ == '__main__':
    main()
