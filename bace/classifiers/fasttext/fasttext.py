from os.path import join as path_join
import fasttext as ft
from bace.classifiers.classifier import Classifier
from typing import List, Text, Tuple


class FastText(Classifier):
    def read_data(self, filename: Text) -> Tuple[List[Text], List[Text]]:
        #dataset = pd.read_csv(filename)
        #return dataset['label'].tolist(), dataset['tokens'].tolist()
        raise NotImplementedError()

    def get_classifier(self, train_labels: List[Text], train_data: List[Text], num: int = 500):
        #return ft.supervised(train)
        raise NotImplementedError()

    def predict(self, classifier, test_data: List[Text], num: int = 200):
        #return classifier.predict(CountVectorizer(max_features=num).fit_transform(test_data).toarray())
        raise NotImplementedError()

def main():
    train_file_name = path_join("filtered_data", "fasttext_train.txt")
    test_file_name = path_join("filtered_data", "fasttext_test.txt")

    classifier = ft.supervised(train_file_name, "model")

    results = classifier.test(test_file_name)
    print(results.precision)
    print(results.recall)

    prediction = classifier.predict(["convict"], k=3)
    print(prediction)


if __name__ == "__main__":
    main()