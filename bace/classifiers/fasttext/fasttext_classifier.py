from os.path import join as path_join
from os.path import abspath as abs_path
from os.path import exists as path_exists
from os import makedirs as make_dirs
# from fasttext import supervised, train_supervised
import fasttext as ft


def run_fasttext(args):
    def fancy_print_predictions(classifier, input_str: str, k: int):
        predictions = classifier.predict_proba(
            [input_str], k=k
        )
        for curr_string_prediction in predictions:
            for label, prob in curr_string_prediction:
                print(label, ": ", prob, sep='')

    # print(args)

    train_file_name = abs_path(
        path_join(args.input_dir, args.train_file)
    )
    test_file_name = abs_path(
        path_join(args.input_dir, args.test_file)
    )

    classifier = ft.supervised(train_file_name, args.binary_file)
    results = classifier.test(test_file_name)

    if args.metrics:
        print("Number of test slices:", results.nexamples)
        print("Precision:", results.precision)
        print("Recall:", results.recall)
        # print()
        # fancy_print_predictions(
        #     classifier,
        #     "robbed",
        #     len(classifier.labels)
        # )
    else:
        if not path_exists(abs_path(args.output_dir)):
            make_dirs(abs_path(args.output_dir))

        with open(
            abs_path(path_join(args.output_dir, args.metrics_file)),
            "w"
        ) as f:
            f.write(f"Number of test slices: {results.nexamples}\n")
            f.write(f"Precision: {results.precision}\n")
            f.write(f"Recall: {results.recall}\n")


def construct_parser_fasttext(subparser):
    """
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
    """

    subparser.add_argument(
        'input_dir', type=str, default="data_clean", metavar="input-dir",
        help='Input directory to preprocessed data'
    )
    subparser.add_argument(
        '-o', '--output-dir', type=str, default="results",
        help='Output directory to hold fasttext classifier output files'
    )
    subparser.add_argument(
        '--train_file', type=str, default="fasttext_train.txt",
        help='Name of training file found in "--input-dir"'
    )
    subparser.add_argument(
        '--test_file', type=str, default="fasttext_test.txt",
        help='Name of testing file found in "--input-dir"'
    )
    subparser.add_argument(
        '-b', '--binary_file', type=str, default="model",
        help='Name of fasttext model binary that will be generated.'
    )

    # Make results showing options mutually exclusive
    fasttext_results = subparser.add_mutually_exclusive_group()
    fasttext_results.add_argument(
        '-m', '--metrics', action="store_true", default=False,
        help="Flag to enable/disable showing fasttext metrics to command line"
    )
    fasttext_results.add_argument(
        '--metrics-file', type=str, default="fasttext_metrics.txt",
        help='Filename to store fasttext metrics'
    )

    subparser.set_defaults(run=run_fasttext)


def main():
    train_file_name = path_join("filtered_data_clean", "fasttext_train.txt")
    test_file_name = path_join("filtered_data_clean", "fasttext_test.txt")

    classifier = ft.supervised(train_file_name, "model")

    results = classifier.test(test_file_name)
    print(results.precision)
    print(results.recall)

    prediction = classifier.predict(["convict"], k=3)
    print(prediction)

if __name__ == "__main__":
    main()
