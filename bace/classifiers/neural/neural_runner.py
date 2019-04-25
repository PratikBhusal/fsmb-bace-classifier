
def run_nn(args):
    raise NotImplementedError()

def construct_parser_nn(subparser):
    subparser.add_argument(
        'training_folder', type=str, default="data", metavar="input-dir",
        help='Path to input training data'
    )
    subparser.add_argument(
        '-g', '--glove_embedding', type=str,
        default="data", metavar="input-dir",
        help='Path to glove embedding file and its number of dimensions'
    )

    subparser.add_argument(
        '-o', '--output_dir', type=str, default="results",
        help='Output directory to hold bow classifier output files'
    )
    subparser.add_argument(
        '-t', '--test_folder', type=str, default="test_texts.csv",
        help='Directory containing directories containing text files'
    )
    subparser.add_argument(
        '-n', '--num_features', type=int, default=200,
        help='number of features to use in bag of words classification'
    )

    # Make results showing options mutually exclusive
    subparser.add_argument(
        '-m', '--metrics', action="store_true", default=False,
        help="Flag to just show metrics instead of predictions"
    )

    subparser.set_defaults(run=run_nn)