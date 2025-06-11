# Conversation-Inference-Tree

## What is this package for?
This package was made to allow a user to generate a summarized explanation of any provided reddit thread, by applying a series of user-created questions in a predetermined logical sequence.  The intended use case is to allow the user to extract the underlying reasoning of conversations about a given subject at a massive scale, then save those inferences to create a dataset or report. A more detailed explanation of how the package logic operates can be found [here](docs/overview.md).

## Installing the package
### pip install
*Pending publishing of package*

### git clone
```bash
Need to retrieve github https
cd Conversation-Inference-Tree
```
## Basic Usage Example
```python
from Conversation_Inference_Tree.inference_tree import InferenceTree
import json

#Load the data you want to get a summary of.
with open("scraped_reddit_thread.json", "r", encoding="utf-8") as f:
    thread = json.load(f)

#Instantiate the InferenceTree object.  Make sure to do this outside of any loops!
inference_object = InferenceTree
#Set the llm that will be used for generation by inference_object's internal logic.
inference_object.set_llm(model_name="meta-llama/Llama-3.2-3B-Instruct", model_type="hf")

#Set the summarizer and the agents.
#NOTE: Functionality pending

#Trigger inference
summary = inference_object.process_thread(data=thread, data_type="json")

print(summary)
```

## Liscense
*Need a liscense*

## Documentation
- [Overview](docs/overview.md)
- [Usage\ Examples](docs/usage.md)
- [API\ Reference](docs/api.md)