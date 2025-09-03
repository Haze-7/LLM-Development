#LLM Devlopment HW 1:

import argparse
import random


# from collections import OrderedDict

#Pseudo Code: 

#1. Read text file provided 

"""
#2. Write a python class for bigram / trigram model that can be trained on a string of data:
  a. class __init__ method :
        class Model: 
            def __init__(n) :
                function work
                init instance vars here
                n is int (2 or 3 based on trigram or bigram)

            def train() :
                function work
                accepts one positional argument (a string of data for the model to be trained on)
                split (.split() or re.split()) string into seperate words (punctuation are seperate words (learn more))
                identify unique words in corpus (?)
                identify which other words can follow unique word (?)
                qualitfy the probability that each word follows other words (calc from formula / get %/ chance)
            
            def predict_next_word(input, deterministic) :
                function work
                Args:
                    input: 
                        -tuple
                        -contains one or two prior words ( 1 for bigram, 2 for trigram)
                    deterministic:
                        -Boolean Flag
                        -defaults to False
                        -use input to sample the next word from probability distribution from training (pick highest % prob from above in order)
                        if True:
                            - always sample HIGHEST probability next word (greedy sampling ^^)
                        if False:
                            -randomly sample the token using probability distribution itself (categorial sampling) (?) 
                        ^^(Tip: use random.choices function(import random) for sampling (used in rock paper scissors game))
                
                -must detect when user asks to run predict on word that isn't in vocabulary/ print out error message (input validation)
                (maybe use re.findall or re.search (if false, give error message))
"""

"""
#3: Command Line Interface w/ argparse library

 Idea: allow the program to be interfaced / run from the terminal in custom fashion (allow traversal / selection)

  a. (required) argument
   -first positional (required) a arg should be a selector for which activity to perform
   - options:
    1. train_ngram 
    2. predict_ngram

  b. (--data) argument
    - that points to the PATH(including filename and extension) to training data 
    - only used if user select (train_ngram_ activity)

  c. (--save) Argument
    - points to the path where ngram or BPE model (depending on activity chosen) will be saved
    -^ do so it can be loaded in predict_ngram activity (save training data to then convert / show in predictions)
    - can be serialized and saved using pickle

  d. (--load) Argument
    - points to the PATH where the TRAINED ngram model was saved (above --save)
    - model object can be loaded using pickle (learn)

  e. (--word) String Argument
    - specifies the first word / words used for predict_ngram activity (?)

  f. (--nwords) Int Argument
    - specifies the # of words to predict for the predict_ngram activity
    - (# of words to include in what is next likely i assume)

  g. (--n) Int Argument
    - specifies the order of the ngram 
    - (choices: 1 or 2)
    -only needed for train_ngram activity

  h. (--d) Argument 
    - set the deterministic flag for the next predict_next_word() method of ngram model to True
    - only used for predict_ngram activity
"""

"""
Plan out / think through the two activities:

Activities:

  1. train_ngram :
    - Runs training of model

    Uses:
      -train() method
      (--data) arg
      (--n) arg

  2. predict_ngram:
    - Runs prediction alg(calc?) / display to user

    Uses:
      -predict_next_word() method
      (--save) arg
      (--word) arg
      (--nwords) arg
      (--d) arg

"""

"""
4. Make sure to have proper comments / documentation 
(check standard)
Add template here to use for rest of code: 


"""

#2. Tokenize it with .split (identify each word) / w/ for loop
#3. add ^ into ordered set (no dupes, indexes) / key: value pair w/ index: value (string)
#4. return the set / set is tokens
#5.  Do probablility / trigram / bigram formuala to calc frequency. 
#   Higher the freq, goes first, then in order


