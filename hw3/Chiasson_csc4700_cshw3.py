"""
LLM Development HW 3:

Project Description

"""
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time # for tracking sleep function

#load environmental variables
load_dotenv('.env')

key = os.getenv("OPENAI_API_KEY")

#establish client
client = OpenAI(api_key = key)

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

class APIModels():

    def __init__(self):
        self.client = OpenAI(api_key = key)

    def get_questions(self, dataset_path = "dev-v2.0.json", limit = 500):

        #load dataset
        with open(dataset_path, "r") as file:
            dataset = json.load(file)

        questions = [] #list of questions pulled from dataset

        for entry in dataset["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    if not qa.get("is_impossible", False): #skip if is_impossible == False
                        questions.append({"id": qa["id"], "question": qa["question"] }) #add entry to new list
                        if len(questions) >= limit:
                            return questions
        return questions # in case dataset is smaller than 500 entries (maybe get rid of )

    def gpt5_nano_batch(self, limit = 500):
        
        #retreieve question set from get_questions method
        question_set = self.get_questions(limit = limit)

        print(f"Number of questions in batch: {len(question_set)}")


        #NEXT, prepare batch file
        with open("batch_input_file", "w") as file:
            for question in question_set:
                line = {
                    "custom_id": question["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions", #may change
                    "body": {
                        "model": "gpt-5-nano",
                        #"reasoning_effort": "minimal",
                        "messages": [
                            {"role": "developer", "content": "Your job is to take in questions and provide answers to them. When offered a multiple choice, select the correct choice."}, #Update with proper question
                            {"role": "user", "content": question["question"]},
                        ]
                    }
                }
                file.write(json.dumps(line) + "\n") #dump / output (change later)


        #Next, upload .jsonl file to openAi/ store file ID
        #client = OpenAI()

        batch_file = self.client.files.create(
            file = open("batch_input_file", "rb"),
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

        #Next, create batch job
        #load file_id from tracker file
        with open(tracker_file, "r") as track_file:
            tracker_data = json.load(track_file)

        #setup / useable for batch job
        input_file_id = tracker_data["input_file_id"]

        #now, create batch job
        batch_job = self.client.batches.create(
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
        batch_job = self.client.batches.retrieve(batch_job.id)
        batch_status = batch_job.status # get status for selected batch job ( by id)
        print("Batch Job Status:", batch_status) #or just batch?

        last_status = None

        #make periodic with while loop (sleep for 60 seconds)
        while True:

            batch_job = self.client.batches.retrieve(batch_job.id) #update batch job status
            batch_status = batch_job.status

            if batch_status != last_status:
                print ("Batch Job Status:", batch_status)
                

            if batch_status == "completed":
                output_file_id = batch_job.output_file_id

                if not output_file_id:
                    print("Batch job completed, waiting for output File...")
                    time.sleep(20)
                    continue 

                print("Batch job completed successfully.") # move on from here, may drop to end
                #get results / output

                output_file_id = batch_job.output_file_id # get id of output file for batch job (field)
                batch_results = self.client.files.content(output_file_id) #use ^ to get output file content (batch results)

                #save to local file (suggested by documentation)

                if hasattr(batch_results, "read"):
                    batch_results_data = batch_results.read()
                else:
                    batch_results_data = bytes(batch_results)

                with open("batch_results.jsonl", "wb") as batch_results_file:
                    batch_results_file.write(batch_results_data)
                        
                #get question text (create dictionary from original set to get question text from ID) (for display purposes)
                id_to_question = {q["id"]: q["question"] for q in question_set}

                #finally, parse results file to get answers
                results = []

                with open("batch_results.jsonl", "r") as batch_results_file:
                    for line in batch_results_file:
                        result_line = (json.loads(line)) #each line of results file (get data)
                        question_id = result_line.get("custom_id")
                        
                        #handle multiple choice questions / answers errors
                        try:
                            answer = result_line["response"]["body"]["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError):
                            answer = None


                        question_text = id_to_question.get(question_id)

                        results.append({
                            "id": question_id, 
                            "question": question_text,
                            "answer": answer
                        })

                    #output nicely?:
                    for result in results:
                        print(f"Question ID: {result['id']}\nQuestion: {result['question']}\nAnswer: {result['answer']}\n")

            elif batch_status == "failed":
                print("Batch job failed. Please check the details.")
            elif batch_status == "cancelled":
                print("Batch job was cancelled.")
            elif batch_status == "expired":
                print("Batch job ran out of time and expired.")
            else: 
                print("Batch job is still in progress. Current status:", batch_status)
                
            time.sleep(20)  # Sleep for 20 seconds before checking again

def main():
        API = APIModels()
        API.gpt5_nano_batch(limit = 3)
    # APIModels.gpt5_nano_batch()

if __name__ == "__main__":
    main()




            #link to docs
            #current: https://platform.openai.com/docs/guides/batch

            #from class:
            #https://platform.openai.com/docs/api-reference/batch/create
            #https://platform.openai.com/docs/overview