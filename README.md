**FSMB BACE Classifier**

This repository is for a board action classifier for the federation of state medical boards. This is a senior design 
project by UTD computer science undergraduate students that will use natural-language processing and machine learning 
techniques to classify documents based on limited training data.

The training and testing data will be in the form of text files that are cleaned as much as possible after being 
extracted from the PDF documents that the various medical boards provides via OCR techniques.

The goal of this project is to, given some input training data, use various techniques to classify new data.

**Notable Files**

* bace_driver.py : The main driver for the various functionalities, including the preprocessor and the classifiers.
* bace/ : Modules that bace_driver.py depends upon 
    * classifiers/ :
    * preprocessor.py :