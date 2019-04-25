from bace.classifiers.bayesian.bag_of_words import construct_parser_bow
from bace.classifiers.fasttext.fasttext_classifier import construct_parser_fasttext
from bace.classifiers.neural.neural_runner import construct_parser_nn
from bace.preprocessor import construct_parser_preprocessor
import argparse


def construct_primary_parser():
    def subparser_constructors():
        yield 'pp', construct_parser_preprocessor
        yield 'ft', construct_parser_fasttext
        yield 'bow', construct_parser_bow
        yield 'nn', construct_parser_nn

    parser = argparse.ArgumentParser(description='Classify documents and \
                                     subsections using various NLP techniques')

    # Create framework for for preprocessor and classifier(s) frameworks
    subparsers = parser.add_subparsers(help='pp for preprocessor, ft for fasttext, \
                                        bow for bag of words, nn for neural',
                                       dest='task')
    subparsers.required = True

    # construct_parner_nn(subparsers.add_parser("nn"))
    for name, cons in subparser_constructors():
        new_subparser = subparsers.add_parser(name)
        cons(new_subparser)

    return parser

def main():
    args = construct_primary_parser().parse_args()
    args.run(args)

if __name__ == "__main__":
    main()