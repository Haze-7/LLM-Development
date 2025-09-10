#LLM Devlopment HW 2:

import argparse
import pickle
import random
import re

#Steps:
"""
1. read text file provided (reuse)
2. Python Class BPE Alg:
    Args:

        vocabulary:
         -class attribute
         -named self.vocabulary

         train() method:
            args:
                trainign data corpus 
                optional integer arg k:
                    default to 500
            -implement BPE learner alg to create token vocab
            - k = # of iterations fo the BPE loop to run

        tokenize() method:
            args:
                single string arg (text to tokenize)
            
            -use trained vocab to tokenize probiced string
            return a Tuple containing 2 items:
                1. tokens
                2. token IDs that correspond to ^^


3. Command Line Interface:

required:
    -first positional arg
    -selector for which activity to perform (reuse)
    options : train_bpe / tokenize

--data:
    -poitns to path to training corpus
    -only use in train (reuse)

--save:
    points to path where model is saved (model.p)
    -reuse from last
    -save w/ pickle
    -load in tookenize
    Model: "datastructure that defines vocab"

--load:
    -points to path of trained model (model.p)
    - model object load w/ oickle

--text: (string arg)
    - specifies the string to be tokenized in tokenize method
"""

class BPEAlgorithm:
    """
    Method explanation
    """

    def __innit__(self):
        """
        Method explain
        Arguments:
            1.
            2.
        """
        self.vocabulary = set()

    def train(self, corpus, k = 500):
        """
        Method Explain

        Arguments:
            corpus (str): Text data to train model on.
            k (int, optional): Number of iterations of the BPE loop to run. Defaults to 500.
        """
        pass

    def tokenize(self, text):
        """
        Method Explain

        Arguments:
            text(str) = Text to be tokenized.

        Returns:
            Tuple: Contains tokens and their corresponding tokenIDs.
        """

        
        pass

        

def main():
    """
    Main function, handles Command Line Interface for training and running n-gram model, decisions, pathing, and final output prints
    """
    
    # Command Line Interface
    parser = argparse.ArgumentParser(description = "BPE Algorithm")

    #Activity Selector
    parser.add_argument("activity", type=str, choices=["train_bpe", "tokenize"], help = "Select Activity to preform on Algorithm.")

    #b. An argument (--data) that points to the path of the training data corpus. Only use for train_bpe activity
    parser.add_argument("--data", help = "Poits to the path of the training corpus.")

    #c. An argument (--save) that points to the path where the BPE algorithm will be saved to be loaded in tokenize activity (model.p).
    parser.add_argument("--save", help = "Path to the path where trained Algorithm data will be saved.")

    #d. An argument (--load) that poitns to the path where the BPE algorithm was saved (model.p)
    parser.add_argument("--load", help = "Points to the path where trained algorithm was saved.")

    #e. An string argument (--text) that specifies the string to be tokenized in tokenize activity.
    parser.add_argument("--text", type = str, help = "Specifies the string to be tokenized in tokenize activity.")

    #Activities Decision Handling
    args = parser.parse_args()

    if args.activity == "train_bpe":
        corpus_path = args.data
        text = args.text # may need to move to tokenize / sets strign to be used in tokenize so most likely here
        save_path = args.save

        with open(corpus_path, "r", encoding="UTF-8") as file:
            corpus = file.read()

        algorithm = BPEAlgorithm() 
        algorithm.train(corpus) # don't need to pass k in here if i don't want to

        #save training data ^^
        if save_path:
            with open(save_path, "wb") as file:
                pickle.dump(algorithm, file)

    elif args.activity == "tokenize":
        load_algorithm = args.load
        
        if not load_algorithm:
            raise ValueError("Error: Missing Load/Training File")

        #retrieve / load training file
        with open(load_algorithm, "rb") as algorithm_file:
            pickle.load(algorithm_file)

        #handle the rest of the solving (hard part), depends on BPE model training above




if __name__ == "__main__":
    main()