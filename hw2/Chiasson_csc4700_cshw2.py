"""
LLM Development HW 2:

This program implements a Byte Pair Encoding algorithm to tokenize a given string of text
and outputs the subword tokens learned from training, as well as a recreation of the provided
string using new, learned vocabulary. 

"""

import argparse
import pickle

class BPEAlgorithm:
    """
    Byte Pair Encoding Algorithm that tokenizes a given corpus of text.
    """

    def __init__(self):
        """
        Initialize algorithm with set of unique vocabulary, a list of the vocabulary(which
        is later sorted to determine merge order), a dictionary for mapping tokens to their
        token IDs, and another dictionary for mapping those token IDs back into their tokens.
        Arguments:
            None
        """
        self.vocabulary = set()
        self.vocabulary_list = []
        self.convert_tokens_to_ids = {}
        self.convert_ids_to_tokens = {}
        """
        Coding Note:
        After noticing my code took a long time to train, I asked AI to identify
        places where I could optimize.
        AI suggested I permiate the vocab list & the conversion mappings throughout 
        the project, as I previously created vocabulary list both in main() and in 
        tokenize(). With this new method, I instead create them all at once within train().
        I then use them to convert the tokens to their token ID's within the 
        tokenize() method, and then to convert those tokenId's back to their
        tokens within main() to reduce repetition.
        """

    def train(self, corpus, k = 500):
        """
        Creates a vocabulary from provided corpus text by tracking and documenting frequently 
        adjacent tokens to create a more comprehensive / efficient set to tokenize text. 
        Accuracy / Efficiency improved with larger k (more opportunities to merge).

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

        self.vocabulary_list = sorted(list(self.vocabulary))
        self.convert_tokens_to_ids = {t: i for i, t in enumerate(self.vocabulary_list)}
        self.convert_ids_to_tokens = {i: t for i, t in enumerate(self.vocabulary_list)}

        return self.vocabulary

    def tokenize(self, text):
        """
        Uses trained vocabulary to tokenize corpus text by grabbing individual characters and merging
        them into pairs based on the set of frequently adjacent pairs established while training.
        
        Arguments:
            text(str) = Text to be tokenized.

        Returns:
            Tuple: Contains tokens and their corresponding tokenIDs.
        """
        #turn into list
        tokens = list(text)

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

        #map tokens to their id's
        token_ids = [self.convert_tokens_to_ids[t] for t in tokens]

        return tokens, token_ids


def main():
    """
    Main function that handles user interaction with a dedicated Command Line Interface using Argparse.
    Handles recieving user input, calling necessary functions, and converting token ids back to tokens
    (characters) to be printed.
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

    #added extra, as it was required by the autograder
    parser.add_argument("--k", type = int, default = 500, help = "Sets the number of BPE merge operations.")

    #Activities Decision Handling
    args = parser.parse_args()

    if args.activity == "train_bpe":
        corpus_path = args.data
        save_path = args.save
        num_merges = args.k

        with open(corpus_path, "r", encoding="UTF-8") as file:
            corpus = file.read()

        algorithm = BPEAlgorithm() 
        algorithm.train(corpus, num_merges)

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

        #bring tokens back from Ids
        reconstruct_tokens = [algorithm.convert_ids_to_tokens[tokens] for tokens in token_ids]

        #rejoin tokens into string:
        reconstruct_text = "".join(reconstruct_tokens)
        
        print(reconstruct_tokens) #list of tokens (to show merge pairs.)
        print(reconstruct_text) # text reverted back to its initial string (to show proper reading/ conversion)

        #Train Example Command:
        #python3 Chiasson_csc4700_cshw2.py train_bpe --data corpus.txt --save model.p

        #Tokenize Example Command:
        #python3 Chiasson_csc4700_cshw2.py tokenize --text 'The bright green Norwegian avocado was eaten by the whale!' --load model.p

if __name__ == "__main__":
    main()