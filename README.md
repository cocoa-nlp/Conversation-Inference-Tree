# Conversation-Inference-Tree

## What is this package for?
This package was made to allow a user to generate a summarized explanation of any provided reddit thread, by applying a series of user-created questions in a predetermined logical sequence.  The intended use case is to allow the user to extract the underlying reasoning of conversations about a given subject at a massive scale, then save those inferences to create a dataset or report. A more detailed explanation of how the package logic operates can be found [here](docs/overview.md).

## Installing the package
### pip install
*Pending publishing of package*

### git clone
```bash
git clone https://github.com/cocoa-nlp/Conversation-Inference-Tree.git
cd Conversation-Inference-Tree
```
## Basic Usage Example
```python
from Conversation_Inference_Tree.inference_tree import InferenceTree
import json
import os

#Retrieve the data to be processed from its json file
with open("post_1m0350d.json", "r", encoding="utf-8") as f:
    thread = json.load(f)

#Define the model id to be used
model = "meta-llama/Llama-3.1-8B-Instruct"

#Initialize the object(outside of any loops)
inference_object = InferenceTree(model, "hf") 

#Process the conversation(inside loop if processing multiple conversations)
summary = inference_object.process_thread(thread, data_type="json")

#Save summary
for count, s in summary:
    text_file = open(f"{thread["id"]}_Report{count}.txt", "w")
    text_file.write(s)
    text_file.close()
```

## Liscense
*Need a liscense*

## Documentation
- [Overview](docs/overview.md)
- [Usage\ Examples](ConversationInferenceTree/docs/usage.md)