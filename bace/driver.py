from   bace.classifiers.bayesian import bag_of_words as bow
from   bace.classifiers.fasttext import fasttext     as ft
from   os.path                   import join         as path_join
import argparse

def construct_parser():
    parser = argparse.ArgumentParser(description='Classify documents and subsections using various NLP techniques')
    # parser.add_argument('data', type=str, help='training file directory containing a subfolder per class')
    # parser.add_argument('test', type=str, help='test file or directory')
    parser.add_argument('data', type=str, help='folder containing input data directories')
    parser.add_argument('-c' '--classifier', type=str, default='bow', required=True, help='used classifier from {bow '
                                                                                          '(bag of '
                                                                             'words), '
                                                                   'fasttext}')
    parser.add_argument('-n', '--numfeatures', type=int, default=200, help='number of features to use in '
                                                              'classification')
    parser.add_argument('-s', '--slicedata', action='store_true', help='output sliced results')
    parser.add_argument('-r', '--raw', type=str, help='input data is raw, will be preprocessed then output to given '
                                                      'directory')

    return parser

def get_classifier_types():
    return {
        "bow" : bow,
        "fasttext" : ft
    }

def main():
    args = construct_parser().parse_args()

    num_feats = 200 if args.numfeatures == None else args.numfeatures
    train_file_name = path_join("filtered_data","train_texts.csv")
    test_file_name = path_join("filtered_data","test_texts.csv")

    clf_type = get_classifier_types()[args.classifier]

    labels_train, data_train = clf_type.read_data(train_file_name)
    labels_test, data_test = clf_type.read_data(test_file_name)
    clf = clf_type.get_classifier(labels_train, data_train, num_feats)
    pred_test = clf_type.predict(clf, data_test)

    for i in range(len(labels_test)):
        print(labels_test[i] == pred_test[i], labels_test[i], pred_test[i], "|||", data_test[i])

if __name__ == "__main__":
    main()