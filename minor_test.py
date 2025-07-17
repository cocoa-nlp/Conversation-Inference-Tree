from Conversation_Inference_Tree.inference_tree import InferenceTree
import json
import os

#Temporary test code space
#load json files into list with name "threads"
with open("posts/post_1m0v50d.json", "r", encoding="utf-8") as f:
    thread = json.load(f)

model = "meta-llama/Llama-3.1-8B-Instruct"

#Threads stands for the list of RedditWrapper objects
inference_object = InferenceTree(model, "hf", graph=True) 
summary = inference_object.process_thread(thread, data_type="json")

text_file = open("/home/umflint.edu/brayclou/Conversation-Inference-Tree/Conversation_Inference_Tree/summaries/" + thread["id"] + ".txt", "w")
text_file.write(summary)
text_file.close()