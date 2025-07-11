from Conversation_Inference_Tree.inference_tree import InferenceTree
import json
import os


#Temporary test code space
#load json files into list with name "threads"
with open("posts/post_1l3aknm.json", "r", encoding="utf-8") as f:
    thread = json.load(f)


model = "prithivMLmods/Acrux-500M-o1-Journey"
q_list = [
    {
        "question": "Summarize the text in 150 words or less.",
        "depth": -99
    },
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

#Temporary print for testing purposes
print(summary)
exit()

text_file = open("summaries/" + thread["id"] + ".txt", "w")
text_file.write(summary)
text_file.close()