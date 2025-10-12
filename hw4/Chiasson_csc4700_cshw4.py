#HW 4
"""
LLM Development HW 4:


"""
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time # for tracking sleep function
import requests
import argparse

#load environmental variables
load_dotenv('.env')

key = os.getenv("OPENAI_API_KEY")
router_key = os.getenv("OPENROUTER_API_KEY")

#establish client
client = OpenAI(api_key = key)



#Extract first 500 questions/ answers
#if == is_impossible, skip


#iterate through, get file

class APIModels():

    def __init__(self):
        self.client = OpenAI(api_key = key)

    def get_questions(self, dataset_path = "dev-v2.0.json", limit = 500):
        """
        Traverse / Convert data set to workable .jsonl file for batch/ other api functionality

        Arguments:
        dataset_path: The path / name of the dataset to be used/analyzed.
        limit: # of questions to consider (excluding impossible questions) 
        """

        #load dataset
        with open(dataset_path, "r") as file:
            dataset = json.load(file)

        questions = [] #list of questions pulled from dataset

        for entry in dataset["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    if not qa.get("is_impossible", False): #skip if is_impossible == False
                        #extract correct answers:
                        answers = [answer["text"] for answer in qa.get("answers", []) if answer.get("text")]

                        questions.append({
                            "id": qa["id"], 
                            "question": qa["question"],
                            "answers": answers
                        }) #add entry to new list
                        if len(questions) >= limit:
                            return questions
        return questions # in case dataset is smaller than 500 entries (maybe get rid of )

    def openai_batch(self, limit = 500):
        """
        Run API call to gpt-5-nano as a student answering questions in batch configuration.

        Arguments:
        limit: Sets the # of questions to read (excluding is_impossible).

        Source Used:
        https://platform.openai.com/docs/guides/batch
        """
        
        #retreieve question set from get_questions method
        question_set = self.get_questions(limit = limit)

        print(f"Number of questions in batch: {len(question_set)}")

        openai_batch_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "student_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "object",
                            "description": "Student's answer to the given question."
                        },
                    },
                    "required": ["answer"],
                }
            }
        }

        #prepare batch file
        with open("batch_input_file", "w") as file:
            for question in question_set:
                line = {
                    "custom_id": question["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions", #may change
                    "body": {
                        "model": "gpt-5-nano",
                        "reasoning_effort": "minimal",
                        "messages": [
                            {"role": "developer", "content": "Your job is to take in questions and provide answers to them. When offered a multiple choice, select the correct choice."},
                            {"role": "user", "content": question["question"]},
                        ],
                        "response_format": openai_batch_schema
                    }
                }
                file.write(json.dumps(line) + "\n") #dump / output (change later)


        #Next, upload .jsonl file to openAi/ store file ID
        #client = OpenAI()

        batch_file = self.client.files.create(
            file = open("batch_input_file", "rb"),
            purpose = "batch"
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


        #track / check batch job status: (initial / first time (may remove / redundant with loop below))
        # batch_job = self.client.batches.retrieve(batch_job.id)
        # batch_status = batch_job.status # get status for selected batch job ( by id)
        # print("Batch Job Status:", batch_status) #or just batch?

        last_status = None

        #make periodic with while loop (sleep for 60 seconds)
        while True:

            batch_job = self.client.batches.retrieve(batch_job.id) #update batch job status
            batch_status = batch_job.status

            if batch_status != last_status:
                print ("Batch Job Status:", batch_status)
                last_status = batch_status
                

            if batch_status == "completed":
                output_file_id = batch_job.output_file_id

                if not output_file_id:
                    print("Batch job completed, waiting for output File...")
                    time.sleep(20)
                    continue 
                          
                print("Batch job completed successfully.") # move on from here, may drop to end
                #get results / output
                #start edit call
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
                        result_line = json.loads(line)
                        question_id = result_line.get("custom_id")


                        # Extract the answer from choices[0].message.content
                        #answer_json = result_line.get("response", {}).get("body", {}).get("choices", [])[0].get("message", {}).get("content", "{}")
                        answer_json = (
                            result_line.get("response", {}).get("body", {}).get("choices", [])[0].get("message", {}).get("content", "{}")
                        )
                        try:
                            answer_data = json.loads(answer_json)
                            #student_answer = answer_data.get("answer", {}).get("value") or answer_data.get("answer", {}).get("description")
                            answer_field = answer_data.get("answer", {})
                            student_answer = (
                                answer_field.get("description")  # base questions
                                or answer_field.get("content")   # multiple choice fallback
                            )
                        except json.JSONDecodeError:
                            student_answer = answer_json

                        # Preserve question text from original question set
                        question_text = id_to_question.get(question_id)
                        
                        results.append({
                            "id": question_id,
                            "question": question_text,
                            "answer": student_answer
                        })

                    #output nicely?:
                    for result in results:
                        #print(f"Question ID: {result['id']}\nQuestion: {result['question']}\nAnswer: {result['answer']}\n")
                        print(f"Question ID: {result['id']}\nQuestion: {result['question']}\nAnswer: {result['answer']}\n")

                    break #exit while loop / end program

            elif batch_status == "failed":
                print("Batch job failed. Please check the details.")
            elif batch_status == "cancelled":
                print("Batch job was cancelled.")
            elif batch_status == "expired":
                print("Batch job ran out of time and expired.")

            time.sleep(20)  # Sleep for 20 seconds before checking again


    def openrouter_serial(self, limit = 500):
        """
        Run API call to qwen3/ qwen3-8b as a student answering questions in serial configuration.

        Arguments:
        limit: Sets the # of questions to read (excluding is_impossible).
        """

        questions = self.get_questions(limit = limit)

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {router_key}",
            "Content-type": "application/json",
        }

        results = []
        
        #create file / reset each run (so it doesn't continue to pile up (error I was having))
        with open("openrouter_results.jsonl", "w") as file:
            pass

        for idx, question in enumerate(questions[:limit], 1):
            body = {
                "model": "qwen/qwen3-8b",
                "messages": [
                    { "role": "system", "content": "Your job is to take in questions and provide answers to them. Answer each question with only the final concise answer. Do not provide explanations, context, or markdown formatting."}, 
                    { "role": "user", "content": question["question"]}
                ]
            }

            try:
                response = requests.post(url, headers = headers, json = body)
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, requests.exceptions.RequestException):
                print(f"Error on question")
                answer = None

            results.append({
                "id": question["id"],
                "question": question["question"],
                "answer": answer
            })

            #append to file
            with open("openrouter_results.jsonl", "a") as file:
                file.write(json.dumps(results[-1]) + "\n")
 
            print(f"Processed question {idx}/{min(limit, len(questions))}")

        for result in results:
            print(f"Question ID: {result['id']}\nQuestion: {result['question']}\nAnswer: {result['answer']}\n")

    
        #break #exit while loop / end program


    def openai_grader(self, limit = 500):
        """
        Grader with gpt-5-mini batch mode to score responses of previous models(students).
        Provides a score (True or False), then gives an explanation as for why

        Arguments:
        limit: Sets the # of questions to read (excluding is_impossible).
        
        """
        #get/ convert questions from dataset
        questions = self.get_questions(limit = limit)

        #store question ids in dictionaries
        id_to_question = {question["id"]: question["question"] for question in questions}
        id_to_correct_answers = {question["id"]: question["answers"] for question in questions}
        

        #result files from batch (gpt-5-nano) & streaming (qwen3-8b)
        model_results = {
            "gpt-5-nano": "batch_results.jsonl",
            "qwen3-8b": "openrouter_results.jsonl"
        }

        grading_response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "grader_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "boolean",
                            "description": "Boolean decision if student answer is correct(True) or incorrect(False)"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "A short explanation of why student answer was or wasn't correct."
                        }
                    },
                    "required": ["score", "explanation"],
                    "additionalProperties": False
                }
            }
        }

        #prepare batch input file
        with open("grader_input_file", "w") as file:
            for model, result_file in model_results.items():
                #read each model results 
                with open(result_file, "r") as grader_file: #return here
                    for line in grader_file:
                        result_data = json.loads(line)
                        question_id = result_data['id']
                        question = id_to_question.get(question_id, "Question Missing")
                        student_response = result_data.get("answer", "")
                        
                        correct_answers = id_to_correct_answers.get(question_id, [])

                        grading_prompt = f"""
                            You are a teacher tasked with determining whether a student’s answer to a question was correct,
                            based on a set of possible correct answers. You must only use the provided possible correct answers
                            to determine if the student’s response was correct.
                        
                            Question: {question}
                            Student’s Response: {student_response}
                            Possible Correct Answers:
                            {correct_answers}
                            Your response should only be a valid Json as shown below:
                            {{
                            "explanation" (str): A short explanation of why the student’s answer was correct or
                            incorrect.,
                            "score" (bool): true if the student’s answer was correct, false if it was incorrect
                            }}
                            Your response: 
                            """
                        
                        #create batchline for gpt-5 mini

                        batch_line = {
                            "custom_id": f"{model}_{question_id}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": "gpt-5-mini",
                                "messages": [
                                    {"role": "system", "content": "You are a teacher grading students answers to questions and providing appropriate explanations. Strictly follow the user's prompts"},
                                    {"role": "user", "content": grading_prompt}
                                ],
                                "response_format": grading_response_schema
                            }
                        }

                        #write line to grader input file:
                        file.write(json.dumps(batch_line) + "\n")
                        
            #upload batch file to use with openAI
            grader_batch_file = self.client.files.create(
                file = open("grader_input_file", "rb"),
                purpose = "batch"
            )

            grader_file_id = grader_batch_file.id

            #create tracker file
            grader_tracker_file = "grader_tracker_file.json"
            grader_tracker_data = {"input_file_id": grader_file_id}

            #write to file
            with open (grader_tracker_file, "w") as grader_track_file:
                json.dump(grader_tracker_data, grader_track_file, indent = 2)

            with open(grader_tracker_file, "r") as grader_track_file:
                grader_tracker_data = json.load(grader_track_file)
            
            #setup / usable for batch job
            grader_file_id = grader_tracker_data["input_file_id"]

            #create batch job
            grader_batch_job = self.client.batches.create(
                input_file_id = grader_file_id,
                endpoint = "/v1/chat/completions",
                completion_window="24h",
                metadata = {
                    "description": "GPT-5 Mini HW3 Grader Batch Job"
                }
            )

            #save batch id to batch job
            grader_tracker_data["grader_batch_job_id"] = grader_batch_job.id

            #write to / update tracker file
            with open(grader_tracker_file, "w") as grader_batch_track_file:
                json.dump(grader_tracker_data, grader_batch_track_file, indent = 2)

            #track / check job status
            grader_batch_job = self.client.batches.retrieve(grader_batch_job.id)
            grader_batch_status = grader_batch_job.status
            # print("Grader batch Status:", grader_batch_status)

            last_status = None

            #track status loop
            while True:
                grader_batch_job = self.client.batches.retrieve(grader_batch_job.id)
                grader_batch_status = grader_batch_job.status

                if grader_batch_status != last_status:
                    print("Grader Batch Job Status:", grader_batch_status)
                    last_status = grader_batch_status
                
                #handle if complete
                if grader_batch_status == "completed":
                    grader_output_file_id = grader_batch_job.output_file_id

                    if not grader_output_file_id:
                        print("Grading Batch Job completed successfully, waiting for output file...")
                        time.sleep(20)
                        continue
                    
                    print("Grading Batch job completed successfully.")

                    #get results / output
                    grader_output_file_id = grader_batch_job.output_file_id
                    grader_batch_results = self.client.files.content(grader_output_file_id)

                    #save to local file(may not need)
                    if hasattr(grader_batch_results, "read"):
                        grader_batch_results_data = grader_batch_results.read()
                    else:
                        grader_batch_results_data = bytes(grader_batch_results)

                    #save locally
                    with open("grader_batch_results.jsonl", "wb") as grader_batch_results_file:
                        grader_batch_results_file.write(grader_batch_results_data)
                    
                    #parse results file for output data
                    results = []

                    with open("grader_batch_results.jsonl", "r") as grader_batch_results_file:
                        for line in grader_batch_results_file:
                            result_line = json.loads(line)
                            question_id = result_line.get("custom_id")
                            # Grab the response body
                            body = result_line.get("response", {}).get("body", {})

                            try:
                                # Get the content string from the first choice
                                message_content = (
                                    body.get("choices", [{}])[0]
                                    .get("message", {})
                                    .get("content", "")
                                    .strip()
                                )

                                # Parse the JSON string from the model
                                structured_output = json.loads(message_content)
                                score = structured_output.get("score")
                                explanation = structured_output.get("explanation")

                            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                                score = None
                                explanation = None

                            results.append({
                                "id": question_id,
                                "score": score,
                                "explanation": explanation
                            })

                        for result in results:
                            print(f"Question ID: {result['id']}\nScore: {result['score']}\nExplanation: {result['explanation']}\n")
                        
                        break
                elif grader_batch_status == "failed":
                    print("Batch job failed. Please check the details.")
                elif grader_batch_status == "cancelled":
                    print("Batch job was cancelled.")
                elif grader_batch_status == "expired":
                    print("Batch job ran out of time and expired.")

                time.sleep(20)  # Sleep for 20 seconds before checking again



                        
def main():
    API = APIModels()

    # Command Line Interface
    parser = argparse.ArgumentParser(description = "API Interaction")

    #Define args
    #Activity Selector
    parser.add_argument("activity", type=str, choices=["openai_batch", "openrouter_serial", "openai_grader"], help = "Select Model / Format to run for answering questions.")

    #b. An argument (--limit) that sets the limit / number of questions to read
    parser.add_argument("--limit", type = int, default = 500, help = "Sets the number of questions (excluding impossible) to process.")

    
    args = parser.parse_args()

    limit = args.limit
    # Handle activity choices running
    if args.activity == "openai_batch":
        API.openai_batch(limit)
    elif args.activity == "openrouter_serial":
        API.openrouter_serial(limit)
    elif args.activity == "openai_grader":
        API.openai_grader(limit)

if __name__ == "__main__":
    main()