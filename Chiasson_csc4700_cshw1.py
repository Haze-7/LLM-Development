#LLM Devlopment HW 1:

import argparse
import pickle
import random
import re

class Model :
   """N-Gram Language model with Bi-gram and Tri-gram functionality. """

    
   def __init__(self, n):
    """
    Initialize model with n-gram order.

    Arguments:
      n (int): Must be 2 (bigram) or 3 (trigram).

    """
    if n != 2 and n != 3 :
      raise ValueError("Value of n must equal to 2 or 3.")
    else :
      self.n = n

    self.unique_tokens = set() 
    self.frequency = {} 
    self.probabilities = {}
 

   def train(self, corpus):
    """
    Train n-gram model on provided text corpus.
    
    Arguments:
      corpus (str): Input text for training.
    """
    tokens = re.findall(r"\w+|[^\w\s]", corpus.lower())
   
    self.unique_tokens = set(tokens) 

    if self.n == 2 :
      for i in range (len(tokens) - 1):
        context = tokens[i] 
        next_word = tokens[i + 1] 

        if context not in self.frequency:
          self.frequency[context] = {}

        if next_word not in self.frequency[context]:
          self.frequency[context][next_word] = 0 
        
        self.frequency[context][next_word] += 1 

        word_count = self.frequency[context][next_word]

        sum_count = sum(self.frequency[context].values())

        if context not in self.probabilities :
          self.probabilities[context] = {}

        self.probabilities[context][next_word] = word_count / sum_count

    if self.n == 3 :
      for i in range (len(tokens) - 2):
        context = (tokens[i], tokens[i + 1]) 
        next_word = tokens[i + 2]

        if context not in self.frequency:
          self.frequency[context] = {}

       
        if next_word not in self.frequency[context]:
          self.frequency[context][next_word] = 0 
        
        self.frequency[context][next_word] += 1 

        word_count = self.frequency[context][next_word]

        sum_count = sum(self.frequency[context].values())

        if context not in self.probabilities :
          self.probabilities[context] = {}

        self.probabilities[context][next_word] = word_count / sum_count


   def predict_next_word(self, input, deterministic=False):
    """
    Predict the next word to follow context based on data gathered from training n-gram model.
    
    Arguments:
      input (tuple): Contains one (bigram) or two (trigram) prior words for predictions.
      deterministic (boolean flag): Determines if the prediction samples the highest probability (greedy sampling) 
                                    or if it randomly sasmples a token using the probability distribution 
                                    (categorical sampling). Defaults to false.
    """
    if self.n == 2:
      if len(input) < 1:
        print("Error Message: Need at least 1 word of context to run Bigram model!")
      elif not all(word in self.unique_tokens for word in input):
        print("Error Message: One or more words are not found within training Corpus. Please try again.")
      else :
        context = input[-1] 
        next_probability = self.probabilities.get(context, {}) 

        if not next_probability:
          print("Error Message: No predictions for this context.")

        if deterministic:
          next_word = max(next_probability, key = lambda word: next_probability[word]) 
          return next_word
        else : 
          words = list(next_probability.keys()) 
          probabilities = list(next_probability.values())

          next_word = random.choices(words, weights = probabilities, k = 1)[0] #
          return next_word

    if self.n == 3:
      if len(input) < 2:
        print("Error Message: Need at least 2 word of context to run Trigram model!")
      elif not all(word in self.unique_tokens for word in input):
        print("Error Message: One or more words are not found within training Corpus. Please try again.")
      else :
        context = (input[-2], input[-1]) 
        next_probability = self.probabilities.get(context, {})

        if not next_probability :
          print("Error Message: No predictions for this context.")

        if deterministic :
          next_word = max(next_probability, key= next_probability.get)
          return next_word
        else : 
          words = list(next_probability.keys())
          probabilities = list(next_probability.values())
          next_word = random.choices(words, weights = probabilities, k=1)[0]
          return next_word
    

def main() :
  """Command Line Interface for training and running n-gram model """

  parser = argparse.ArgumentParser(description = "N-Gram Language Model")

  #a. Activity Selector
  parser.add_argument("activity", type=str, choices = ["train_ngram", "predict_ngram"], help = "Select Activity to perform on Model.")

  #b. An argument (--data) that points to the path of training corpus.
  parser.add_argument("--data", help = "Points to path of training corpus.")   

  #c. An argument (--save) that points to the path where the ngram or BPE model will be saved so that it can be loaded in 
  parser.add_argument("--save", help = "Path to where Model is saved to be loaded.")

  #d. An argument (--load) that points to the path where the trained ngram model was saved.
  parser.add_argument("--load", help = "Path to where trained ngram model is saved.")

  #e. A string argument (--word) that specifies the first word (or words) used for the “predict_ngram” activity.
  parser.add_argument("--word", help = "Specifies the first word(or words) for the Prediction activity.")

  #f. An integer argument (--nwords) that specifies the number of words to predict for the “predict_ngram” activity.
  parser.add_argument("--nwords", type = int, help = "Specifies the number of words to predict for the Prediction activity.")

  #g. An integer arugment (--n) that specifies the order of the ngram (choices should be 2 or 3). 
  parser.add_argument("--n", type = int, choices = [1, 2], help = "Select Order of the Ngram 2 (bi) or 3 (tri).")

  #h. An argument (--d) that set the deterministic flag for the predict_next_word() methodof of the ngram model to True.
  parser.add_argument("--d", action = "store_true", help = "Set the deterministic flag for Prediction Model.")

  #Activities Handling:
  args = parser.parse_args()

  if args.activity == "train_ngram":
    
    corpus_path = args.data 
    n_order = args.n
    save_path = args.save 

    with open(corpus_path, "r", encoding="UTF-8") as file:
      corpus = file.read()

    model = Model(n = n_order)
    model.train(corpus)

    if save_path :
      with open(save_path, "wb") as file:
        pickle.dump(model, file)

  elif args.activity == "predict_ngram":
    context_words = args.word 
    num_words = args.nwords 
    deterministic = args.d 
    load_model = args.load 

    if not load_model :
      raise ValueError("Error: Missing load file")

    with open(load_model, "rb") as model_file:
      model = pickle.load(model_file) 

    prediction = []
    current_context = re.findall(r"w+|[^\w\s]", context_words.lower())

    for _ in range(num_words): 
      next_word = model.predict_next_word(current_context, deterministic) 

      prediction.append(next_word)
      current_context.append(next_word)

      if model.n == 2:
        current_context = current_context[-1:] 
      elif model.n == 3:
        current_context = current_context[-2:] 

    print("Prediction:", " ".join(prediction))
    

if __name__ == "__main__":
    main()
