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
import chromadb
from chromadb.utils import embedding_functions

from datetime import datetime
#load environmental variables
load_dotenv('.env')

key = os.getenv("OPENAI_API_KEY")
router_key = os.getenv("OPENROUTER_API_KEY")

#establish client
client = OpenAI(api_key = key)
class APIModels():
    """
    Handles LLM API interactions with RAG capabilities.
    """

    def __init__(self):
        """
        Initialize OpenAI client and ChromaDB with embeddings.
        """
        self.client = OpenAI(api_key = key)

        self.chroma_client = chromadb.PersistentClient(path="./kb")

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key = os.getenv("OPENAI_API_KEY"),
            model_name = "text-embedding-3-small"
        )

    def chunk_text(self, text, chunk_size = 500, overlap = 0):
        """
        Split text into chunks

        Arguments:
        text: Text that we will be chunking
        chunk_size: # of characters per chunk (default to 512)
        overlap: Number of overlapping characters between neighboring chunks (default to 0)
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        step = chunk_size - overlap 
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]#set starting positions, extract


    def chromadb_setup(self, dataset_path = "dev-v2.0.json", chunk_size = 512, overlap = 50):
        """
        Extract contents of dataset, chunk them, and store in chromaDB

        Arguments:
        dataset_path: The path / name of the dataset to be used/analyzed.
        chunk_size: # of characters per chunk (default to 512)
        overlap: Number of overlapping characters between neighboring chunks (default to 50)
        """
        #load dataset
        with open(dataset_path, "r") as file:
            dataset = json.load(file)

        #get / create Chroma DB collection        
        collection = self.chroma_client.get_or_create_collection(
            name ="dataset_contexts", 
            embedding_function = self.embedding_function
        )

        #track unique ids & total # of chunks
        chunk_id = 0
        total_chunks = 0

        #extract all context from full dataset/ chunk
        for entry in dataset["data"]:
            title = entry.get("title", "Unknown Title") #topic title of question

            for paragraph in entry["paragraphs"]:
                context = paragraph["context"]

                #skip / don't store questions w/o context in chroma DB database
                #reason: no need to, just use surrounding / other questions context instead
                if not context:
                    continue

                #get all questions ids from paragraph
                question_ids = [qa["id"] for qa in paragraph["qas"]]
                
                #chunk context
                chunks = self.chunk_text(context, chunk_size, overlap)

                #store chunks in chromaDB w/ upsert
                for chunk in chunks:
                  collection.upsert(
                      ids = [f"chunk_{chunk_id}"],
                      documents = [chunk],
                      metadatas = [{
                          "title": title,
                          "question_ids": json.dumps(question_ids),
                          "full_context": context #full context before chunking to compare/trace
                      }]
                  )
                  chunk_id += 1 #iterate to keep unique
                  total_chunks += 1 #keep track /update total # of chunks
        print(f"ChromaDB setup complete. Stored {total_chunks} chunks from all contexts.")
        return collection


    def retrieve_context(self, question, n_results = 5):
        """
        Retrieve / gather relevant chroma chunks for a question from ChromaDB

        Arguments:
        question: The question were finding context for
        n_results: # of relevant chunks to retrieve / gather (default to 5)

        Returns:
        Combined string of retrieved context chunks
        """
        #get collection of contexts
        collection = self.chroma_client.get_collection(
            name = "dataset_contexts",
            embedding_function = self.embedding_function
        )

        #perform vector semantic search
        results = collection.query(
            query_texts = [question],
            n_results = n_results
        )

        #combine retrieved chunks
        retrieved_contexts = results['documents'][0]
        combined_context = "\n\n".join(retrieved_contexts)

        return combined_context        


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
                context = paragraph.get("context", "")
                for qa in paragraph["qas"]:
                    if not qa.get("is_impossible", False): #skip if is_impossible == False
                        #extract correct answers:
                        answers = [answer["text"] for answer in qa.get("answers", []) if answer.get("text")]

                        questions.append({
                            "id": qa["id"], 
                            "question": qa["question"],
                            "answers": answers,
                            "context": context #include even if empty for now
                        }) #add entry to new list

                        if len(questions) >= limit:
                            return questions
        return questions


    def openai_batch(self, limit = 500, use_rag = True):
        """
        Run API call to gpt-5-nano as a student answering questions in batch configuration.

        Arguments:
        limit: Sets the # of questions to read (excluding is_impossible).
        use_rag: Boolean decision on if to use RAG(get context before answering)

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
                            "type": "string", #changed
                            "description": "Student's answer to the given question."
                        },
                    },
                    "required": ["answer"],
                }
            }
        }

        #prepare batch file (now with RAG)
        with open("batch_input_file", "w") as file:
            for question in question_set:
                #rag decision
                if use_rag:
                    context = self.retrieve_context(question["question"], n_results = 5)
                    prompt = f"Context: {context}\n\nQuestion: {question['question']}\n\nAnswer based on the provided context:"
                else:
                    prompt = question["question"]

                line = {
                    "custom_id": question["id"],
                    "method": "POST",
                    "url": "/v1/chat/completions", #may change
                    "body": {
                        "model": "gpt-5-nano",
                        "reasoning_effort": "minimal",
                        "messages": [
                            {"role": "developer", "content": "Your job is to take in questions and provide concise answers to them based on the provided context."},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": openai_batch_schema
                    }
                }
                file.write(json.dumps(line) + "\n") #dump / output (change later)


        #Next, upload .jsonl file to openAi
        batch_file = self.client.files.create(
            file = open("batch_input_file", "rb"),
            purpose = "batch"
        )
        #get back file id to track
        file_id = batch_file.id

        #create / setup tracker file
        tracker_file = "tracker_file.json"
        #add file id to tracker file
        tracker_data = {"input_file_id": file_id}

        #write to file
        with open (tracker_file, "w") as track_file:
            json.dump(tracker_data, track_file, indent = 2)

        #create batch job
        with open(tracker_file, "r") as track_file:
            tracker_data = json.load(track_file)

        #setup / useable for batch job
        input_file_id = tracker_data["input_file_id"]

        #submit batch job to OpenAI
        batch_job = self.client.batches.create(
            input_file_id = input_file_id,
            endpoint="/v1/chat/completions",
            completion_window = "24h",
            metadata = {
                "description": "GPT-5 Nano HW4 Batch Job with RAG"
            }
        )
        tracker_data["batch_job_id"] = batch_job.id

        #write to / update tracker file
        with open (tracker_file, "w") as track_file:
            json.dump(tracker_data, track_file, indent = 2)


        #track status
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
                          
                print("Batch job completed successfully.")
                batch_results = self.client.files.content(output_file_id) 

                #save to local file (suggested by documentation)
                if hasattr(batch_results, "read"):
                    batch_results_data = batch_results.read()
                else:
                    batch_results_data = bytes(batch_results)

                with open("batch_results.jsonl", "wb") as batch_results_file:
                    batch_results_file.write(batch_results_data)
                        
                #get question text (create dictionary from original set to get question text from ID) (for display purposes)
                id_to_question = {q["id"]: q["question"] for q in question_set}

                results = []

                with open("batch_results.jsonl", "r") as batch_results_file:
                    for line in batch_results_file:
                        result_line = json.loads(line)
                        question_id = result_line.get("custom_id")

                        answer_json = (
                            result_line.get("response", {}).get("body", {}).get("choices", [])[0].get("message", {}).get("content", "{}")
                        )
                        try:
                            answer_data = json.loads(answer_json)
                            student_answer = answer_data.get("answer", "")
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
                        print(f"Question ID: {result['id']}\nQuestion: {result['question']}\nAnswer: {result['answer']}\n")

                    break #exit while loop / end program

            elif batch_status == "failed":
                print("Batch job failed. Please check the details.")
                break
            elif batch_status == "cancelled":
                print("Batch job was cancelled.")
                break
            elif batch_status == "expired":
                print("Batch job ran out of time and expired.")
                break

            time.sleep(20)  # Sleep for 20 seconds before checking again


    def openrouter_serial(self, limit = 500, use_rag = True):
        """
        Run API call to qwen3/ qwen3-8b as a student answering questions in serial configuration, now with RAG.

        Arguments:
        limit: Sets the # of questions to read (excluding is_impossible).
        use_rag: Boolean decision on if to use RAG(get context before answering)
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
            
            if use_rag:
                context = self.retrieve_context(question["question"], n_results = 5)
                prompt = f"Context: {context}\n\nQuestion: {question['question']}\n\nAnswer based on the provided context:"
            else:
                prompt = question["question"]

            body = {
                "model": "qwen/qwen3-8b",
                "messages": [
                    { "role": "system", "content": "Your job is to take in questions and provide concise answers based on the provided context. Answer with only the final answer, no explanations."}, 
                    { "role": "user", "content": prompt}
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
            
            #progress tracking
            print(f"Processed question {idx}/{min(limit, len(questions))}")

        for result in results:
            print(f"Question ID: {result['id']}\nQuestion: {result['question']}\nAnswer: {result['answer']}\n")


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
        

        #set which result files to grade / attach/ link to their models
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

                        #debug output:
                        # print(f"DEBUG - Question ID: {question_id}")
                        # print(f"DEBUG - Correct answers: {correct_answers}")
                        # print(f"DEBUG - Student response: {student_response}")
                        # print("---")

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
            
            #create batch job
            grader_batch_job = self.client.batches.create(
                input_file_id = grader_file_id,
                endpoint = "/v1/chat/completions",
                completion_window="24h",
                metadata = {
                    "description": "GPT-5 Mini HW4 Grader Batch Job"
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

                    #track modelscores for final percentages
                    model_scores = {"gpt-5-nano": [], "quen3-8b": []}
                    model_detailed_results = {"gpt-5-nano": [], "quen3-8b": []} #detailed results by model for output files

                    with open("grader_batch_results.jsonl", "r") as grader_batch_results_file:
                        for line in grader_batch_results_file:
                            result_line = json.loads(line)
                            full_id = result_line.get("custom_id")

                            #parse model name / question id
                            if full_id.startswith("gpt-5-nano_"):
                                model_name = "gpt-5-nano"
                                question_id_from_grader = full_id.replace("gpt-5-nano_", "")
                            elif full_id.startswith("qwen3-8b_"):
                                model_name = "qwen3-8b"
                                question_id_from_grader = full_id.replace("qwen3-8b_", "")
                            else:
                                print(f"Unknown model in ID: {full_id}")
                                continue

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

                                #track scores by model
                                if score is not None:
                                    model_scores[model_name].append(1 if score else 0) #convert boolean score to integer
                                    model_detailed_results[model_name].append({
                                        "id": full_id,
                                        "score": score,
                                        "explanation": explanation
                                    })

                            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                                score = None
                                explanation = None

                            results.append({
                                "id": full_id,
                                "score": score,
                                "explanation": explanation
                            })
                            
                        for result in results:
                            print(f"Question ID: {result['id']}\nScore: {result['score']}\nExplanation: {result['explanation']}\n")
                        
                        #save output files
                        current_date = datetime.now().strftime("%Y-%m-%d")

                        for model_name, detailed_results in model_detailed_results.items():
                            output_filename = f"{model_name}-RAG-{current_date}-hw4.json"
                            with open(output_filename, "w") as output_file:
                                json.dump(detailed_results, output_file, indent = 2)
                            print(f"Saved {model_name} results to {output_filename}")
                        
                        #calc / print scores / stats
                        print("Grading Summary")
                        for model_name, scores in model_scores.items():
                            if scores:
                                total = len(scores)
                                correct = sum(scores) #get all scores(bool) where answer is true (correct)
                                percentage = (correct / total) * 100
                                print(f"\n{model_name.upper()}:")
                                print(f"  Total Questions: {total}")
                                print(f"  Correct Answers: {correct}")
                                print(f"  Incorrect Answers: {total - correct}")
                                print(f"  Average Score (Accuracy): {percentage:.2f}%")

                elif grader_batch_status == "failed":
                    print("Batch job failed. Please check the details.")
                    break
                elif grader_batch_status == "cancelled":
                    print("Batch job was cancelled.")
                    break
                elif grader_batch_status == "expired":
                    print("Batch job ran out of time and expired.")
                    break

                time.sleep(20)  # Sleep for 20 seconds before checking again



                        
def main():
    API = APIModels()

    # Command Line Interface
    parser = argparse.ArgumentParser(description = "API Interaction with dataset using RAG")

    #Define args
    #Activity Selector
    parser.add_argument("activity", type=str, choices=["chromadb_setup", "openai_batch", "openrouter_serial", "openai_grader"], help = "Select Model / Format to run for answering questions.")
    #An argument (--limit) that sets the limit / number of questions to read
    parser.add_argument("--limit", type = int, default = 500, help = "Sets the number of questions (excluding impossible) to process.")
    #An argument (--no_rag) that sets if rag will be used or not(if added, skips / doesnt use rag)
    parser.add_argument("--no_rag", action = "store_true", help = "Disable Rag(don't retrieve context)")
    #An argument (--chunk_size) that sets size of text chunks for chromaDB
    parser.add_argument("--chunk-size", type = int, default = 500, help = "Size of text chunks for ChromaDB (default: 500)")
    #An argument (--overlap) that sets size of overlap between chunks
    parser.add_argument("--overlap", type = int, default = 50, help = "Overlap between chunks (default: 50)")
    
    args = parser.parse_args()

    limit = args.limit
    use_rag = not args.no_rag
    chunk_size = args.chunk_size
    overlap = args.overlap

    # Handle activity choices running
    if args.activity == "chromadb_setup":
        API.chromadb_setup(chunk_size = chunk_size, overlap = overlap)
    elif args.activity == "openai_batch":
        API.openai_batch(limit, use_rag)
    elif args.activity == "openrouter_serial":
        API.openrouter_serial(limit, use_rag)
    elif args.activity == "openai_grader":
        API.openai_grader(limit)

if __name__ == "__main__":
    main()


#Steps / Run Procedure:
"""
# Step 1: Setup ChromaDB (run once)
python Chiasson_csc4700_cshw4.py chromadb_setup

# Step 2: Get GPT-5-nano answers with RAG
python Chiasson_csc4700_cshw4.py openai_batch

# Step 3: Get qwen3-8b answers with RAG
python Chiasson_csc4700_cshw4.py openrouter_serial

# Step 4: Grade both models
python Chiasson_csc4700_cshw4.py openai_grader

# Optional: Run without RAG
python Chiasson_csc4700_cshw4.py openai_batch --no-rag

# Optional: Custom chunking
python Chiasson_csc4700_cshw4.py setup_db --chunk-size 1000 --overlap 100

"""