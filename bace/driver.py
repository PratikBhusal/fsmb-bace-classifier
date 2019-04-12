from bace.classifiers.bayesian import bag_of_words        as bow
from bace.classifiers.fasttext import fasttext_classifier as ft
from os.path                   import join                as path_join
import argparse
import bace.preprocessor as pp
import pandas as pd


def construct_parser():
    parser = argparse.ArgumentParser(description='Classify documents and \
                                     subsections using various NLP techniques')

    parser.add_argument('-n', '--numfeatures', type=int, default=200,
                        help='number of features to use in ' 'classification')
    parser.add_argument('-f', '--filter', action="store_true",
                        help='input data is already filtered')
    parser.add_argument('-c', '--classifier', type=str, default='bow',
                             help='The type of trained classifier to generate \
                             a binary version of. Possible Values: \
                             bow (bag of words), fasttext')
    parser.add_argument('-m', '--metrics', action="store_true", default=False,
                        help="Flag to enable/disable showing classifer metrics"
    )
    # parser.add_argument('-o', '--output', type=str,
    #                           help="folder to store program's results.")
    # parser.add_argument('data', type=str, help='training file directory containing a subfolder per class')
    # parser.add_argument('test', type=str, help='test file or directory')

    # input_group = parser.add_mutually_exclusive_group()
    parser.add_argument('-i', '--input', type=str, default="data",
                        required=True,
                        help='folder containing input data.')

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('-s', '--slicedata', type=str,
                              help='output sliced results of a given file')
    output_group.add_argument('-p', '--predict', type=str,
                              help='predict given file for every .txt file in \
                              given directory')

    return parser


def get_classifier_object(class_name: str):
    return {
        "bow": bow,
        "bag of words": bow,
        "fasttext": ft
    }[class_name]


def main():
    args = construct_parser().parse_args()

    classifier_driver = None
    try:
        classifier_driver = get_classifier_object(args.classifier)
    except Exception as e:
        raise e

    if not args.filter:
        # The data is not already filtered. Run through preprocessor.
        train_df, test_df = pp.split_dataset(
            input_dir=args.input,
            should_export_extras=True,
            split_percent=0.8)
        args.input = "filtered_data"

    train_labels, train_tokens = classifier_driver.read_data(
        path_join(args.input, "train_texts.csv")
    )

    classifier = classifier_driver.get_classifier(train_labels, train_tokens)

    test_labels, test_tokens = classifier_driver.read_data(
        path_join(args.input, "train_texts.csv")
    )

    predictions = classifier_driver.predict(classifier, test_tokens)

    if args.metrics:
        classifier_driver.show_metrics(test_labels, predictions)


if __name__ == "__main__":
    main()