import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import bace.classifiers.neural.neural_constants as neural_constants
import bace.classifiers.neural.data_slicer as data_slicer

from keras.models import Sequential
from keras.utils import np_utils
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from bace.classifiers.neural.glove import load_glove

from typing import Dict

import pickle

class NeuralClassifier:

    def __init__(self):
        # a list of tuples of (type, data_clean, true_label)
        self.labelled_data = []
        self.labelled_validation_data = []
        self.model = None
        self.tokenizer = None
        self.labels = []
        self.label_encoder = None
    #force

    def pickle(self, fname, keep_data=False):
        with open(fname, 'w') as f:
            if keep_data:
                pickle.dump(self, f)
            else:
                temp_l_data = self.labelled_data
                temp_v_data = self.labelled_validation_data
                self.labelled_data = []
                self.labelled_validation_data = []
                pickle.dump(self, f)
                self.labelled_data = temp_l_data
                self.labelled_validation_data = temp_v_data
                
    def to_pred(self, pred):
        maxi = 0
        for i in range(1, len(pred)):
            if pred[i] > maxi:
                maxi = i
        return self.labels[maxi]
      
    def to_pred_comparison(self, pred):
        probs = [(self.labels[i], pred[i])for i in range(len(pred))]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs

    def add_data(self, file_id : str, tokenized_file : str, true_label : int):
        """

		:param file_id: a hashable ID for this particular file
		:param tokenized_file: a
		:param true_label:
		:return: None
		"""

        # CURRENTLY NOT TAKING IN PRE-TOKENIZED FILE, DISCUSS WITH TEAM ABOUT ALTERING CLASSIFIER INTERFACE
        if true_label not in self.labels:
            self.labels.append(true_label)
        self.labelled_data.append((file_id, tokenized_file, true_label))

    def add_validation_data(self, file_id : str, data : str, true_label : int):
        """

		:param file_id:
		:param data:
		:param true_label:
		:return:
		"""
        if true_label not in self.labels:
            self.labels.append(true_label)
        self.labelled_validation_data.append((file_id, data, true_label))

    def train(self,
              max_number_tokens=neural_constants.MAX_NUMBER_TOKENS,
              slice_length=neural_constants.SLICE_LENGTH,
              slice_overlap=neural_constants.SLICE_OVERLAP,
              glove_file=neural_constants.GLOVE_FILE,
              glove_dimensions=neural_constants.GLOVE_DIMENSIONS,
              diagnostic_printing=False,
              num_epochs=10,
              batch_size=5):
        """
		This classifier object will train on all the data_clean that has been added to it using the adddata method
		:return:
		"""

        has_validation = len(self.labelled_validation_data) > 0
        # create the tokenizer
        self.tokenizer = Tokenizer(num_words=max_number_tokens)
        training_data = [text for _, text, _ in self.labelled_data]
        self.tokenizer.fit_on_texts(training_data)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        # now build our training data_clean
        X_train = self.tokenizer.texts_to_sequences(training_data)

        if has_validation:
            X_validation = self.tokenizer.texts_to_sequences([text for _, text, _ in self.labelled_validation_data])

        X_train, y_train_labels = data_slicer.slice_data(X_train,
                                                  [y for _, _, y in self.labelled_data],
                                                  slice_length=slice_length,
                                                  overlap_percent=slice_overlap)
        if has_validation:
            X_validation, y_validation_labels = data_slicer.slice_data(X_validation,
                                                            [y for _, _, y in self.labelled_validation_data],
                                                            slice_length=slice_length,
                                                            overlap_percent=slice_overlap)
        # convert labels to 1-hots
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

        y_train = np_utils.to_categorical(self.label_encoder.transform(y_train_labels))
        if has_validation:
            y_validation = np_utils.to_categorical(self.label_encoder.transform(y_validation_labels))


        # pad them as necessary
        if has_validation:
            X_validation = np.array(pad_sequences(X_validation, padding="post", maxlen=slice_length))
        X_train = pad_sequences(X_train, padding="post", maxlen=slice_length)

        # force change

        # get our glove embeddings
        glove = load_glove(glove_file, self.tokenizer.word_index, glove_dimensions)

        # compute some neural_constants
        vocab_size = len(self.tokenizer.word_index) + 1

        # set model parameters
        self.model = Sequential()
        model_layers = [
            # must have these two layers firsts
            layers.Embedding(vocab_size,
                             glove_dimensions,
                             weights=[glove],
                             input_length=slice_length,
                             trainable=False),
            # now we have some options
            layers.GlobalMaxPool1D(),
            layers.Dense(45, activation="relu"),
            layers.Dense(20, activation="sigmoid"),
            # probably want a final sigmoid layer to get smooth value in range (0, 1)
            layers.Dense(len(self.labels), activation="softmax")
        ]
        # add them in
        for layer in model_layers:
            self.model.add(layer)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        """
        print(np.shape(X_train))
        print(np.shape(y_train))
        print(np.shape(X_validation))
        print(np.shape(y_validation))
        """

        #X_train, y_train = shuffle_parallel_arrays(X_train, y_train)

        # now we fit (can take a while)
        if has_validation:
            self.model.fit(X_train, y_train,
                           epochs=num_epochs,
                           verbose=False,
                           shuffle=True,
                           validation_data=(X_validation, y_validation),
                           batch_size=batch_size)
        else:
            self.model.fit(X_train, y_train,
                           epochs=num_epochs,
                           verbose=False,
                           shuffle=True,
                           batch_size=batch_size)

        if neural_constants.DIAGNOSTIC_PRINTING and has_validation:
            def cm(true, pred):
                m = confusion_matrix(true, pred)
                print("Confusion matrix")
                print("   {0:3s} {1:3s}".format("P+", "P-"))
                print("T+ {0:<3d} {1:<3d}".format(m[1][1], m[0][1]))
                print("T- {0:<3d} {1:<3d}".format(m[1][0], m[0][0]))


            y_train_pred = [x for x in list(self.model.predict(X_train, verbose=False))]
            y_validation_pred = [x for x in list(self.model.predict(X_validation, verbose=False))]

            loss, acc = self.model.evaluate(X_train, y_train, verbose=False)
            print("Train L/A asd: {0:.4f} {1:.4f}".format(loss, acc))
           # cm(y_train, y_train_pred)
            loss, acc = self.model.evaluate(X_validation, y_validation, verbose=False)
            print("Validation L/A: {0:.4f} {1:.4f}".format(loss, acc))
            #cm(y_validation, y_validation_pred)

            nc = 0
            for i in range(len(X_validation)):
                print(y_validation_labels[i],self.to_pred(y_validation_pred[i]), y_validation_pred[i])
                if y_validation_labels[i] == self.to_pred(y_validation_pred[i]):
                    nc += 1
            print("acc:", nc/len(y_validation_labels))

    def predict(self, str,
                      slice_length=neural_constants.SLICE_LENGTH,
                      slice_overlap=neural_constants.SLICE_OVERLAP):

        tokenized = self.tokenizer.texts_to_sequences([str])
        slices, _ = data_slicer.slice_data(tokenized,
                                           None,
                                           slice_length=slice_length,
                                           overlap_percent=slice_overlap)
        #print(slices)
        X = np.array(pad_sequences(slices, padding="post", maxlen=slice_length))
        #print(X)
        predictions = [x for x in list(self.model.predict(X, verbose=False))]

        s = predictions[0]
        for p in predictions[1:]:
            for i in range(len(s)):
                s[i] += p[i]
        return self.to_pred_comparison([x / sum(s) for x in s])

    def slice_and_predict(self, str,
                          slice_length=neural_constants.SLICE_LENGTH,
                          slice_overlap=neural_constants.SLICE_OVERLAP):
        tokenized = self.tokenizer.texts_to_sequences([str])
        slices, _ = data_slicer.slice_data(tokenized,
                                        None,
                                        slice_length=slice_length,
                                        overlap_percent=slice_overlap)
        restored = self.tokenizer.sequences_to_texts(slices)
        #print(slices)
        X = np.array(pad_sequences(slices, padding="post", maxlen=slice_length))
        #print(X)
        predictions = [x for x in list(self.model.predict(X, verbose=False))]
        return [
            (self.to_pred(predictions[i]),
             restored[i])
            for i in range(len(slices))]