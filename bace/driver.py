from   bace.classifiers.bayesian import bag_of_words as bow
from   bace.classifiers.fasttext import fasttext     as ft
from   os.path                   import join         as path_join
import argparse

def construct_parser():
    parser = argparse.ArgumentParser(description='Classify documents and subsections using various NLP techniques')
    # parser.add_argument('data', type=str, help='training file directory containing a subfolder per class')
    # parser.add_argument('test', type=str, help='test file or directory')

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-d', '--data', type=str, help='folder containing input data directories')
    input_group.add_argument('-c', '--classifier', type=str, default='bow', help='used classifier from {bow '
                                                                                          '(bag of '
                                                                             'words), '
                                                                   'fasttext}')
    #parser.add_argument('-n', '--numfeatures', type=int, default=200, help='number of features to use in '
    #                                                         'classification')


    parser.add_argument('-r', '--raw', type=str, help='input data is raw, will be preprocessed then output to given '
                                                      'directory')

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('-s', '--slicedata', type=str, help='output sliced results of a given file')
    output_group.add_argument('-p', '--predict', type=str, help='predict given file or every .txt file in given '
                                                                'directory')

    return parser

def main():
    args = construct_parser().parse_args()

    if

if __name__ == "__main__":
    main()