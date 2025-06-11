from Conversation_Inference_Tree.inference_tree import InferenceTree
import json
import os


#Temporary test code space
#load json files into list with name "threads"
file_names = os.listdir("posts")
threads = []
summaries = []
for name in file_names:
    with open("posts/" + name, "r", encoding="utf-8") as f:
        threads.append(json.load(f))


model = "meta-llama/Llama-3.2-3B-Instruct"
#Threads stands for the list of RedditWrapper objects
inference_object = InferenceTree()
inference_object.set_llm(model, "test") #NOTE: real options are either hf or openai
for thread in threads:
    summary = inference_object.process_thread(thread, data_type="json")

    text_file = open("summaries/" + thread["id"] + ".txt", "w")
    text_file.write(summary)
    text_file.close()