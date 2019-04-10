import pandas as pd
from typing import List, Text, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from os.path import join as path_join
import argparse

def read_data(filename: Text) -> Tuple[List[Text], List[Text]]:
    dataset = pd.read_csv(filename)
    return dataset['label'].tolist(), dataset['tokens'].tolist()

def get_classifier(train_labels: List[Text], train_data:List[Text],
              num: int = 500):
    classifier = GaussianNB()
    classifier.fit(CountVectorizer(max_features = num).fit_transform(train_data).toarray(),train_labels)
    return classifier

def predict(classifier: GaussianNB, test_data:List[Text], num: int = 200):
    return classifier.predict(CountVectorizer(max_features = num).fit_transform(test_data).toarray())

def show_metrics(test_labels: List[Text], class_prediction: List[Text]):
    print(confusion_matrix(test_labels,class_prediction))
    print(classification_report(test_labels,class_prediction))
    print(accuracy_score(test_labels,class_prediction))
