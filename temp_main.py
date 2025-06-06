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


model = "llama-3.2-3b-Instruct"
#Threads stands for the list of RedditWrapper objects
inference_object = InferenceTree()
for thread in threads:
    inference_object.set_llm(model, "huggingface")

    #Set agents
    print(f"thread data type: {type(thread)}")
    summary = inference_object.process_thread(thread)

    text_file = open("summaries/" + thread["id"] + ".txt", "w")
    text_file.write(summary)
    text_file.close()