from Conversation_Inference_Tree.inference_tree import InferenceTree
import json
import os


#Temporary test code space
#load json files into list with name "threads"
with open("posts/post_1l3aknm.json", "r", encoding="utf-8") as f:
    thread = json.load(f)


model = "meta-llama/Llama-3.1-8B-Instruct"
q_list = [
    {
        "question": "Give a paragraph conveying some main points contained within this social media comment and its summarized replies.",
        "depth": 0,
        "order": 1
    },
    {
        "question": "Summarize the content of this reply", #NOTE: need to adjust
        "depth": 1
    },
]
#Threads stands for the list of RedditWrapper objects
inference_object = InferenceTree(model, "hf", q_list, graph=True) 
summary = inference_object.process_thread(thread, data_type="json")

text_file = open("/home/umflint.edu/brayclou/Conversation-Inference-Tree/Conversation_Inference_Tree/summaries/" + thread["id"] + ".txt", "w")
text_file.write(summary)
text_file.close()