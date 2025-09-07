#LLM Devlopment HW 1:

import argparse
import pickle
import random
import re


# from collections import OrderedDict

#Pseudo Code: 

#1. Read text file provided 
corpusFile = "" # var for holding file path to ensure dynamic / get val from command line args

open(corpusFile, "r")

corpusFile.close() #be sure to close file at the end / may need to move down to end of class functions / make own method

"""
#2. Write a python class for bigram / trigram model that can be trained on a string of data:
  a. class __init__ method :
        class Model: 
            def __init__(n) :
                function work
                init instance vars here
                n is int (2 or 3 based on trigram or bigram)

            def train(string arg) :
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
#Write a python class for bigram / trigram model that can be trained on a string of data:
class Model :
  def __init__(self, n) :
      # do n validation here
    if n != 2 and n != 3 :
      raise ValueError("Value of n must equal to 2 or 3!")
    else :
      self.n = n

    self.uniqueTokens = set() #Set for tracking unique words in text
    self.frequency = {} #Dictionary tracking frequency of each word / setting key: val pair for calcs
    self.probabilities = {} # Dictionary tracking the chances of word being next (key: val pair )
 
  def train(self, corpus) :
    #self.corpus = open(corpus, "r") This is for opening the file, handle elsewhhere, tthis handles the string gotten from that

    # split provided string into seperate words (using .findall / regex pattern / lowercase)
    tokens = re.findall(r"\w+|[^\w\s]", corpus.lower()) #use when tracking location/ order of words (index of list)
   
   # identify the unique words in the corpus (using set)
    self.uniqueTokens = set(tokens) #use when need unique words
   # identify which other words can follow each unique word, and
   # need to do version for bigram / trigram (1 or 2 prev words)

   #bigram version
    if self.n == 2 :
      #loop/ iterate through length of tokens list, stop 1 from end (for i + 1 / "next" word)
      for i in range (len(tokens) - 1) :
        context = tokens[i] # use current word
        nextWord = tokens[i + 1] # to track next word

        #dictionary handling (tracking frequency of words (has its shown up before, how many times?))
        #add to frequency dict if not already there
        if context not in self.frequency :
          self.frequency[context] = {}

        # add to nextWord count if not already there / start list of following words (at 0, first following word)
        if nextWord not in self.frequency[context] :
          self.frequency[context][nextWord] = 0 # add nextWord to dictionary, set count to 0 to start (more like initializing)
        
        self.frequency[context][nextWord] += 1 # add 1 to count of nextWord occuring after context(repeat if for loop)

       #frequency Calculation
       # top of equation:
        wordCount = self.frequency[context][nextWord]

      #bottom of equation (sum of counts)
        sumCount = sum(self.frequency[context].values())

        if context not in self.probabilities :
          self.probabilities[context] = {}
        #probability equation:
        self.probabilities[context][nextWord] = wordCount / sumCount


    if self.n == 3 :
      #Similar, but stop 2 from end (trigram uses i + 2)
      for i in range (len(tokens) - 2) :
        context = (tokens[i], tokens[i + 1]) # use current and next word
        nextWord = tokens[i + 2] #to check 3rd word (trigram)

        # quantify the probability that each word follows other words / same logic as bigram
        #dictionary handling (tracking frequency of words (has its shown up before, how many times?))
        #add to frequency dict if not already there
        if context not in self.frequency :
          self.frequency[context] = {}

        # add to nextWord count if not already there / start list of following words (at 0, first following word)
        if nextWord not in self.frequency[context] :
          self.frequency[context][nextWord] = 0 # add nextWord to dict, set count to 0 to start (more like initializing)
        
        self.frequency[context][nextWord] += 1 # add 1 to count of nextWord occuring after context(repeat if for loop)

        #Frequency Calculation
       # top of equation:
        wordCount = self.frequency[context][nextWord]

      #bottom of equation (sum of counts)
        sumCount = sum(self.frequency[context].values())

        if context not in self.probabilities :
          self.probabilities[context] = {}
        #probability equation:
        self.probabilities[context][nextWord] = wordCount / sumCount


  def predict_next_word(self, input, deterministic=False) :

    #input validation ( 2 or 3)
    #take in user input, last 1 or 2 words (bi or tri gram)
    # validae to make sure they're inside text, match needed length
    # for bigram ^ just need 1, for trigram, require 2\
    
    #bigram validation
    if self.n == 2 :
      if len(input) < 1 :
        print("Error Message: Need at least 1 word of context to run Bigram model!")
      elif not all(word in self.uniqueTokens for word in input) :
        print("Error Message: One or more words are not found within training Corpus. Please try again.")
      else :
        #prediction setup
        context = input[-1]  #get context (tuple contents)
        nextProbability = self.probabilities.get(context, {}) 

        #Prediction context empty error handling
        if not nextProbability :
          print("Error Message: No predictions for this context.")

        #Prediction Method handling
        #Greedy
        if deterministic:
          nextWord = max(nextProbability, key=lambda word: nextProbability[word]) #find better way to write this
          return nextWord
        else : #random sample 
          words = list(nextProbability.keys()) #creat list of valid words in context (to randomixe w/ .choices)
          probabilities = list(nextProbability.values()) # get probability vals to act as weights

          nextWord = random.choices(words, weights = probabilities, k=1)[0] #
          return nextWord
        #do prediction (other)

    # trigram validation
    if self.n == 3 :
      if len(input) < 2 :
        print("Error Message: Need at least 2 word of context to run Trigram model!")
      elif not all(word in self.uniqueTokens for word in input) :
        print("Error Message: One or more words are not found within training Corpus. Please try again.")
      else :
        context = (input[-2], input[-1]) # get / use both words of input tuple(alr picked last 2 words from string)
        nextProbability = self.probabilities.get(context, {})

        #Prediction context empty error handling (just in case)
        if not nextProbability :
          print("Error Message: No predictions for this context.")

        #Prediction Method handling
        #Greedy
        if deterministic :
          nextWord = max(nextProbability, key=nextProbability.get)
          return nextWord
        else : #Random (weighted)
          words = list(nextProbability.keys())
          probabilities = list(nextProbability.values())
          nextWord = random.choices(words, weights=probabilities, k=1)[0]
          return nextWord
    

def main () :
  parser = argparse.ArgumentParser(descripttion = "N-Gram Language Model") #create parser object

  parser.add_argument("activity", choices=["train", "predict"], help = "Select Activity to perform on Model.")
  

  pass  
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

    I assume ^^ is where we handle the tuple conversion / identifying lst 1(bi) or 2(tri) words in input and converting to tuple

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


