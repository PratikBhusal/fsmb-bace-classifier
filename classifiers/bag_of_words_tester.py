import pandas as pd
from sys import argv
import os
from typing import List, Text, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from os.path import join as path_join
import argparse
import pickle
import pathlib

def read_data(filename: Text) -> Tuple[List[Text], List[Text]]:
    dataset = pd.read_csv(filename)
    return dataset['label'].tolist(), dataset['tokens'].tolist()

def get_classifier(train_labels: List[Text], train_data:List[Text],
              num: int = 500):
    classifier = GaussianNB()
    return classifier.fit(CountVectorizer(max_features = num).fit_transform(train_data).toarray(),train_labels)


def predict(classifier: GaussianNB, test_data:List[Text], num: int = 500):
    return classifier.predict(CountVectorizer(max_features = num).fit_transform(test_data).toarray())

def show_metrics(test_labels: List[Text], class_prediction: List[Text]):
    print(confusion_matrix(test_labels,class_prediction))
    print(classification_report(test_labels,class_prediction))
    print(accuracy_score(test_labels,class_prediction))

def store_model(classifier: GaussianNB):
    with open(path_join(os.getcwd(),"models","bag_of_words_model.pickle"),"wb") as file:
        pickle.dump(classifier,file)

def load_model() -> GaussianNB:
    with open(path_join(os.getcwd(),"models","bag_of_words_model.pickle"), "rb") as file:
        return pickle.load(file)


def main():
    parser = argparse.ArgumentParser(description='Classify documents and subsections using bayesian techniques')
    #parser.add_argument('data', type=str, help='training file directory containing a subfolder per class')
    #parser.add_argument('test', type=str, help='test file or directory')
    parser.add_argument('-n', '--numfeatures', type=int, help='number of features to use in '
                                                                          'classification, default 200')
    parser.add_argument('-s', '--slicedata', action='store_true', help='output sliced results')
    parser.add_argument('-a', '--archive', help='stores the model')

    if not pathlib.Path(path_join(os.getcwd(),"models","bag_of_words_model.pickle")).exists():
        num_feats = 500 if args.numfeatures == None else args.numfeatures
        train_file_name = path_join("filtered_data","train_texts.csv")
        test_file_name = path_join("filtered_data","test_texts.csv")

        labels_train, data_train = read_data(train_file_name)
        labels_test, data_test = read_data(test_file_name)
        clf = get_classifier(labels_train, data_train, num_feats)
    else:
        clf = load_model()

    if args.archive:
        store_model(clf)

    pred_test = predict(clf, data_test)
'''
    for i in range(len(labels_test)):
        print(labels_test[i] == pred_test[i], labels_test[i], pred_test[i], "|||", data_test[i])
'''
if __name__ == "__main__":
    main()