"""
LLM Development HW 3:

Project Description

"""
from openai import OpenAI
from dotenv import load_dotenv
import os

#load environmental variables
load_dotenv('.env')
# Or load_dotenv('../../.env')

key = os.getenv("OPENAI_API_KEY")

#establish client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




"""

Libraries we can use:
- openai
- python-dotenv
-pydantic


Instructions
1. Get dev set
- extract 1st 500 questions / answers
- skip is_impossible questions

2. w/ api Run GPT-5 Nano in batch mode 
- use minimal reasoning_effort
- generate answers to all questions from step 1
- test w. small batch first (3 or 4 lines)

Question 1: What prompt did I use?

3. w/ Api, run qwen / quen3-8b (serially / not batch) 
- 1 question after another
- generate answers to all questions from step 1 (dev set)

Question 2: What prompt did I use?

4. w/ api, use gpt-5-mini in batch mode to score responses of previous 2
- mark as either True(correct) or False(incorrect)
- provide an explanation
- use STRUCTURED OUTPUTS to ensure vlaid output JSON
- save outputs for each model in Json file that can be reused
- files should be named like (gpt-5-nanp-DATE-hw3.json)
- use structured_outputs feature to get both
    - explanation(str)
    -score (boolean)
Use prompt shown belowL

Question 3: What was the total accuracy of GPT-5 nano?
Question 4: What was the total accuracy of quwen/quwn3-8b
Question 5: What are insights about the results (3-4 sentences)

Scoring Prompt:

At bottom of INstructions


"""