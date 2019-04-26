import bace.classifiers.neural.neural_constants as neural_constants
import bace.classifiers.neural.neural_classifier as nc

from os import listdir

import pickle
from random import shuffle

def run_nn(args):
    def child_path(dir, fname):
        return dir + '/' + fname

    if args.load_saved is not None:
        with open(args.load_saved, 'r') as f:
            clf = pickle.load(f)
    elif args.input:
        classes = listdir(args.input)

        ids = [None] * len(classes)
        for i in range(len(classes)):
            directory = child_path(args.input, classes[i])
            files = list(filter(lambda x : x.endswith(".txt"), listdir(directory)))
            ids[i] = (classes[i], directory, files)

        clf = nc.NeuralClassifier()
        for i in range(len(ids)):
            name, dir, init_inputs = ids[i]
            inputs = init_inputs.copy()
            shuffle(inputs)

            if args.evaluate:
                if not (0 < args.evaluate < 1):
                    raise Exception("Evaluation percent must be in range (0, 1)")
                num_training = int(args.evaluate * len(inputs))
            else:
                num_training = len(inputs)
            X_train = inputs[:num_training]
            X_validate = inputs[num_training+1:]

            for fname in X_train:
                with open(child_path(dir, fname), encoding="windows-1252") as f:
                    clf.add_data("{0} {1}".format(name, fname), f.read(), name)
            for fname in X_validate:
                with open(child_path(dir, fname), encoding="windows-1252") as f:
                    clf.add_validation_data("{0} {1}".format(name, fname), f.read(), name)

        clf.train(max_number_tokens=args.num_tokens,
                  glove_file=args.glove_embedding[0],
                  glove_dimensions=int(args.glove_embedding[1]),
                  num_epochs=args.epochs,
                  batch_size=args.batch_size
                  )

    else:
        raise Exception("Missing mandatory input arg - this error should be impossible")


def construct_parser_nn(subparser):
    input = subparser.add_mutually_exclusive_group(required=True)

    input.add_argument(
        '-i', '--input', type=str, metavar="input-dir",
        help='Path to input training data'
    )
    input.add_argument(
        '-l', '--load_saved', type=str,
        help='Path to saved classifier produced with -s option',
    )


    subparser.add_argument(
        '-g', '--glove_embedding', type=str, nargs=2,
        default="glove_embeddings/glove.6B.100d.txt",
        metavar="glove_file",
        help='Path to glove embedding file and its number of dimensions'
    )

    subparser.add_argument(
        '-o', '--output_dir', type=str, default="results",
        help='Output directory to hold bow classifier output files'
    )

    testing = subparser.add_mutually_exclusive_group(required=True)

    testing.add_argument(
        '-t', '--test_folder', type=str,
        help='Directory containing directories containing text files'
    )
    testing.add_argument(
        '-e', '--evaluate', type=float, metavar="(0-1)",
        help='split and evaluate on input folder with the given percent'
    )
    testing.add_argument(
        '-p', '--pickle', type=str,
        help='saves the model to a given file location with pickle'
    )

    subparser.add_argument(
        '-n', '--num_tokens', type=int, default=neural_constants.MAX_NUMBER_TOKENS,
        help='maximum number of tokens, will increase as data increases and number of classes increases'
    )
    subparser.add_argument(
        '--epochs', type=int, default=neural_constants.NUM_EPOCHS,
        help='maximum number of tokens, will increase as data increases and number of classes increases'
    )
    subparser.add_argument(
        '-b', '--batch_size', type=int, default=neural_constants.MAX_NUMBER_TOKENS,
        help='maximum number of tokens, will increase as data increases and number of classes increases'
    )

    # Make results showing options mutually exclusive
    subparser.add_argument(
        '-m', '--metrics', action="store_true", default=False,
        help="Flag to just show metrics instead of predictions"
    )

    subparser.set_defaults(run=run_nn)