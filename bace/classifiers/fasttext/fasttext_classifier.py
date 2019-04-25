from os.path import join as path_join
import fasttext as ft

def run_fasttext(args):
    raise NotImplementedError()

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
        help='Input directory to preprocessed data_clean'
    )
    subparser.add_argument(
        '-o', '--output-dir', type=str, default="results",
        help='Output directory to hold fasttext classifier output files'
    )
    subparser.add_argument(
        '--train_file', type=str, default="train_texts.csv",
        help='Name of training file found in "--input-dir" directory'
    )
    subparser.add_argument(
        '--test_file', type=str, default="test_texts.csv",
        help='Name of testing file found in "--input" directory'
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

