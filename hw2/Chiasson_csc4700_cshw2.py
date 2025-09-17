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
"""

class BPEAlgorithm:
    """
    Method explanation
    """

    def __init__(self):
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
        #individual character tokenizer:
        tokens = []
        for char in corpus: 
            tokens.append(char) # for each character in the text, add to tokens list (in order)

        self.vocabulary = set(tokens) #add trained data to unique set (training data)

        for i in range(k):

            pair_counts = {} # dictionary to track pairs found and their counts
            #identify pairs
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1]) # tuple of adjacent symbols

                #frequency adding (if in pair count, add 1 more, if not yet, create new entry)
                if pair in pair_counts:
                        pair_counts[pair] += 1
                else:
                        pair_counts[pair] = 1

            #find most frequent pair / 2a.
            most_frequent = max(pair_counts, key = pair_counts.get) # get the pairs, and their keys (the actual counts)

            #merge pair to make new vocab entry (combined)
            merged_pair = "".join(most_frequent) #merge into new pair 'a, b' -> "ab"

            #add new pair to end of vocabulary set
            self.vocabulary.add(merged_pair)

            #do replacement of old tokens / new list to then copy to main list
            updated_tokens = []

            j = 0

            while j < len(tokens): #iterate through tokens, added each one to populate new list
                if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == most_frequent: #if space and identify the most frequent pair, do replacement
                    updated_tokens.append(merged_pair) #add new entry (in first of pair spot)
                    j += 2 # skip over second char in pair, so its not included in new list
                else:
                    updated_tokens.append(tokens[j])
                    j += 1 #if no pair, go to next entry in line as normal(and add it to new list)

            tokens = updated_tokens
        
        return self.vocabulary

    def tokenize(self, text):
        """
        Method Explain

        Arguments:
            text(str) = Text to be tokenized.

        Returns:
            Tuple: Contains tokens and their corresponding tokenIDs.
        """
        #Step 1. split into chars
        tokens = list(text)

        #2. try merging based on vocab
        #use trained vocab (returned from train) to tokenize string)
        trained_vocabulary = self.vocabulary

        #merged order tracking:
        for merge_token in sorted(self.vocabulary, key=len, reverse=True): 
            if len(merge_token) == 1: #skip over original single chars
                continue
            else:
                i = 0
                while i < len(tokens):
                    current_slice = tokens[i : i + len(merge_token)]

                    merge_characters = list(merge_token)

                    if current_slice == merge_characters:
                        tokens[i : i + len(merge_token)] = [merge_token]
                        i += 1
                    else:
                        i += 1

        #Step 3. map tokens to ID
        vocabulary_list = sorted(list(trained_vocabulary))
        token_ids = [vocabulary_list.index(t) for t in tokens]

        return tokens, token_ids


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
        text = args.text 
        
        if not load_algorithm:
            raise ValueError("Error: Missing toad/training file.")
        if not text: 
            raise ValueError("Error: Missing text to tokenize.")

        #retrieve / load training file
        with open(load_algorithm, "rb") as algorithm_file:
            algorithm = pickle.load(algorithm_file)

        tokens, token_ids = algorithm.tokenize(text)

        # Convert back to txt
        vocabulary_list = sorted(list(algorithm.vocabulary))
        id_conversion = {i: t for i, t in enumerate(vocabulary_list)}

        #convert IDss back to tokens
        reconstruct_tokens = [id_conversion[token_id] for token_id in token_ids]

        #rejoin tokens into string:
        reconstruct_text = "".join(reconstruct_tokens)
        
        print("Reconstructed Tokens:", reconstruct_tokens)
        print("Reconstructed Text:", reconstruct_text)
        
        # print("Tokens:", tokens)
        # print("Token IDs:", token_ids)

        #Train Example Command:
        #python3 Chiasson_csc4700_cshw2.py train_bpe --data corpus.txt --save model.p

        #Tokenize Example Command:
        #python3 Chiasson_csc4700_cshw2.py tokenize --text 'The bright green Norwegian avocado was eaten by the whale!' --load model.p

if __name__ == "__main__":
    main()