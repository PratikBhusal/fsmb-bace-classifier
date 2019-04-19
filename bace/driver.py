from bace.classifiers.bayesian import bag_of_words        as bow
from bace.classifiers.fasttext import fasttext_classifier as ft
from os.path                   import join                as path_join
import argparse
import bace.preprocessor as pp
import pandas as pd


def construct_preprocessor_parser(subparser=None):
    def within_percent_interval(interval_str: str) -> float:
        interval = float(interval_str)
        if interval < 0 or interval > 1:
            raise argparse.ArgumentTypeError("Input given is out of bounds!")

        return interval

    preprocess_parser = None
    if subparser:
        preprocess_parser = subparser.add_parser(
            "preprocess",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help="Preprocess given dataset",
        )
    else:
        preprocess_parser = argparse.ArgumentParser(
            description='Preprocess given dataset',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    preprocess_parser.add_argument(
        '-o', '--output-dir', type=str, default="filtered_data",
        help='Output directory to hold preprocessed data'
    )
    preprocess_parser.add_argument(
        'input_dir', type=str, default="data", metavar="input-dir",
        help='Input directory to preprocess'
    )

    # Make file generation options mutually exclusive
    # Note, all 3 of the flags appear. However, we only want 1 of them to
    # appear.
    preprocess_parser.add_argument(
        '--export', type=str, default="single",
        choices=["single", "split", "both"],
        help='Indicate whether you only want a single file holding all of the \
        preprocessed data, or both. If "split\" or "both" were chosen, the \
        split is based on "--train-split" or "--test-split" or have the train \
        split be 80%% of the raw data if neither argument was given.'
    )

    train_test_split = preprocess_parser.add_mutually_exclusive_group()
    train_test_split.add_argument(
        '--train-split', type=within_percent_interval, metavar="[0-1]",
        help="Percentage in interval [0,1] of total data going to the \
        training dataset."
    )
    train_test_split.add_argument(
        '--test-split', type=within_percent_interval, metavar="[0-1]",
        help="Percentage in interval [0,1] of total data going to the \
        testing dataset."
    )

    return preprocess_parser


def construct_bow_parser(subparser=None):
    bow_parser = None
    if subparser:
        bow_parser = subparser.add_parser(
            "bow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help="Bag of words Classifier"
        )
    else:
        bow_parser = argparse.ArgumentParser(
            description='Bag of words Classifier',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    bow_parser.add_argument(
        'input_dir', type=str, default="data", metavar="input-dir",
        help='Input directory to preprocessed data'
    )
    bow_parser.add_argument(
        '-o', '--output-dir', type=str, default="results",
        help='Output directory to hold bow classifier output files'
    )
    bow_parser.add_argument(
        '--train_file', type=str, default="bow_train.txt",
        help='Name of training file found in "--input-dir" directory'
    )
    bow_parser.add_argument(
        '--test_file', type=str, default="bow_train.txt",
        help='Name of testing file found in "--input" directory'
    )
    bow_parser.add_argument(
        '-n', '--numfeatures', type=int, default=200,
        help='number of features to use in bag of words classification'
    )

    # Make results showing options mutually exclusive
    bow_results = bow_parser.add_mutually_exclusive_group()
    bow_results.add_argument(
        '-m', '--metrics', action="store_true", default=False,
        help="Flag to enable/disable showing bow metrics to command line"
    )
    bow_results.add_argument(
        '--metrics-file', type=str, default="bow_metrics.txt",
        help='Filename to store bag of words metrics'
    )

    return bow_parser


def construct_fasttext_parser(subparser=None):
    fasttext_parser = None
    if subparser:
        fasttext_parser = subparser.add_parser(
            "fasttext",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help="Fasttext Classifier"
        )
    else:
        fasttext_parser = argparse.ArgumentParser(
            description='Fasttext Classifier',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    fasttext_parser.add_argument(
        'input_dir', type=str, default="data", metavar="input-dir",
        help='Input directory to preprocessed data'
    )
    fasttext_parser.add_argument(
        '-o', '--output-dir', type=str, default="results",
        help='Output directory to hold fasttext classifier output files'
    )
    fasttext_parser.add_argument(
        '--train_file', type=str, default="fasttext_train.txt",
        help='Name of training file found in "--input-dir" directory'
    )
    fasttext_parser.add_argument(
        '--test_file', type=str, default="fasttext_train.txt",
        help='Name of testing file found in "--input" directory'
    )
    fasttext_parser.add_argument(
        '-b', '--binary_file', type=str, default="model",
        help='Name of fasttext model binary that will be generated.'
    )

    # Make results showing options mutually exclusive
    fasttext_results = fasttext_parser.add_mutually_exclusive_group()
    fasttext_results.add_argument(
        '-m', '--metrics', action="store_true", default=False,
        help="Flag to enable/disable showing fasttext metrics to command line"
    )
    fasttext_results.add_argument(
        '--metrics-file', type=str, default="fasttext_metrics.txt",
        help='Filename to store fasttext metrics'
    )

    return fasttext_parser


def construct_parser():
    parser = argparse.ArgumentParser(description='Classify documents and \
                                     subsections using various NLP techniques')

    # Create framework for for preprocessor and classifier(s) frameworks
    subparsers = parser.add_subparsers(dest="task", required=True)
    classifier_subparsers = subparsers.add_parser(
        "classify",
        help="Classify preprocessed data.").add_subparsers(dest="classifier")

    # Preprocessor & its arguments
    construct_preprocessor_parser(subparsers)

    # Construct Bag of words classifier
    construct_bow_parser(classifier_subparsers)

    # Fasttext classifier & arguments
    construct_fasttext_parser(classifier_subparsers)

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

    # This should be where bag of words numfeatures should be coded in, but the fasttext code does not have numfeatures
    # coded in this method...
    classifier = classifier_driver.get_classifier(train_labels, train_tokens)

    test_labels, test_tokens = classifier_driver.read_data(
        path_join(args.input, "train_texts.csv")
    )

    predictions = classifier_driver.predict(classifier, test_tokens)

    if args.metrics:
        classifier_driver.show_metrics(test_labels, predictions)


if __name__ == "__main__":
    main()