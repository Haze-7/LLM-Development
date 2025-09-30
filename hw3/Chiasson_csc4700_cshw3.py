"""
LLM Development HW 3:

Project Description

"""
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

#load environmental variables
load_dotenv('../../.env')
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

#Extract first 500 questions/ answers
#if == is_impossible, skip


#iterate through, get file

class read():

    def __init__():
        pass
    

def main():

#load dataset
    with open("dev-v2.0.json", "r") as file:
        dataset = json.load(file)
    
    def get_questions(dataset):
        questions = [] #list of questions pulled from dataset

        for entry in dataset["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    if not qa.get("is_impossible", False): #skip if is_impossible == False
                        questions.append({"id": qa["id"], "question": qa["question"] }) #add entry to new list
                        if len(questions) >= 500:
                            return questions
        return questions # in case dataset is smaller than 500 entries (maybe get rid of )
    
    #call:
    question_set = get_questions(dataset)

    #NEXT, prepare batch file

    with open("output_file", "w") as file:
        for question in question_set:
            line = {
                "custom_id:": question["id"],
                "method": "POST",
                "url": "v1/chat/completions", #may change
                "body": {
                    "model": "gpt-5-nano",
                    "reasoning": {"effort": "minimal"}, #change later may be different
                    "messages": [
                        {"role": "developer", "content": "Explain Bot Role / Question"}, #Update with proper question
                        {"role": "user", "content": question["question"]},                       
                    ]
                }
            }
            file.write(json.dumps(line) + "\n") #dump / output (change later)


    #Next, upload .jsonl file to openAi/ store file ID
    client = OpenAI()

    batch_file = client.files.create(
        file = open("output_file", "rb"),
        purpose="batch"
    )

    file_id = batch_file.id

    #create/ setup tracker file
    tracker_file = "tracker_file.json"
    #create /set tracker data format/structure
    tracker_data = {"input_file_id": file_id}

    #write to file /
    with open (tracker_file, "w") as track_file:
        json.dump(tracker_data, track_file, indent = 2)

    #print("File ID:", file_id)


    #Next, create batch job
    #load file_id from tracker file
    with open(tracker_file, "r") as track_file:
        tracker_data = json.load(track_file)

    #setup / useable for batch job
    input_file_id = tracker_data["input_file_id"]

    #now, create batch job
    batch_job = client.batches.create(
        input_file_id = input_file_id,
        endpoint="/v1/chat/completions",
        completion_window = "24h",
        metadata = {
            "description": "GPT-5 Nano HW3 Batch Job",
        }
    )

    #include/ add batch job Id in tracker file
    tracker_data["batch_job_id"] = batch_job.id

    #write to / update tracker file
    with open (tracker_file, "w") as track_file:
        json.dump(tracker_data, track_file, indent = 2)


    #track / check batch job status:
    batch = client.batches.retrieve(batch_job.id)
    #print("Batch Job Status:", batch.status) #or just batch?

            #link to docs
            #current: https://platform.openai.com/docs/guides/batch

            #from class:
            #https://platform.openai.com/docs/api-reference/batch/create
            #https://platform.openai.com/docs/overview