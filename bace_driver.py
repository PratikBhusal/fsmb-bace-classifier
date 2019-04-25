from bace.classifiers.bayesian.bag_of_words import construct_parser_bow
from bace.classifiers.fasttext.fasttext_classifier import construct_parser_fasttext
from bace.preprocessor import construct_parser_preprocessor
import argparse

def get_subparser_constructors():
    return [
        ('pp', construct_parser_preprocessor),
        ('ft', construct_parser_fasttext),
        ('bow', construct_parser_bow)
    ]

def construct_primary_parser():
    parser = argparse.ArgumentParser(description='Classify documents and \
                                     subsections using various NLP techniques')

    # Create framework for for preprocessor and classifier(s) frameworks
    subparsers = parser.add_subparsers(help='pp for preprocessor, ft for fasttext, \
                                        bow for bag of words',
                                       dest='task')
    subparsers.required = True

    for name, cons in get_subparser_constructors():
        new_subparser = subparsers.add_parser(name)
        cons(new_subparser)

    return parser

def main():
    args = construct_primary_parser().parse_args()
    args.run(args)

if __name__ == "__main__":
    main()