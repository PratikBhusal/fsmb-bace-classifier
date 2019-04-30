**FSMB BACE Classifier**

This repository is for a board action classifier for the federation of state medical boards. This is a senior design 
project by UTD computer science undergraduate students that will use natural-language processing and machine learning 
techniques to classify documents based on limited training data.

The training and testing data will be in the form of text files that are cleaned as much as possible after being 
extracted from the PDF documents that the various medical boards provides via OCR techniques.

The goal of this project is to, given some input training data, use various techniques to classify new data.

This project has a dependcy on Fasttext, which may require separate installation.

**Notable Files**

* bace_driver.py : The main driver for the various tasks, including the preprocessor and the classifiers.
* bace/ : Modules that bace_driver.py depends upon 
    * classifiers/ : the modules for the classification tasks
        * bayesian/ : the module for a context-insensitive, bag-of-words based, bayesian classifier.
        * fasttext/ : the module for a context-sensitive classifier based on the fasttext library
        * neural/ : the module for a context-sensitive neural solution that uses GLoVE word vectory
        to represent component words
    * preprocessor.py :
* sample_commands.txt : A selection of example commands that assume that you have the files mentioned blow in root:
    bace_data/data/ : A collection of directories containing any text documents
    bace_data/data_clean/ : A collection of directories containing cleaned text documents
    bace_data/nn/ : Much like data_clean, mentioned above, but potentially with files removed
    bace_data/nn_test/ : A directory holding a collection of files taken from the subdirectories of nn    
    
    
**Citations**

*FastText*
    
>A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

*GLoVe*

>Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
>https://nlp.stanford.edu/pubs/glove.pdf