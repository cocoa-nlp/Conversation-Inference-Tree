# ConverationInferenceTree User Guide
## Overview
InferenceTree is a tool designed to automate the analysis and summarization of Reddit discussions using large language models (LLMs). It constructs a tree representation of a Reddit post and its comments, then applies user-defined agents to extract insights and generate summaries at various depths of the conversation. Each agent or summarizer uses a customizable template to format inputs and outputs, making the tool adaptable to different use cases and models.

The class supports input from Reddit API wrappers like PRAW and its json-saved data, and enables scalable, layered summarization and analysis workflows using HuggingFace or OpenAI models.

## How to use this package; the minimum needed
In order to get this package's functionality off the ground, you only need a couple lines of code.  First, in your command line install the package with the following:
```bash
pip install ConversationInferenceTree
```

Then, import the package within your script:
```python
import ConversationInferenceTree
```

With our package installed, lets get some data to try it out on.  Lets also select a model from huggingface and store a pointer to it in a variable #NOTE: is pointer the best term?
```python
import json
import os

with open("test_praw_data.json", "r", encoding="utf-8") as f:
    thread = json.load(f)

model = "meta-llama/Llama-3.1-8B-Instruct"
```

Just like that, we now have all the setup we need before we actually start using the package functionality.  In order to use the package at its most basic, you need two lines; one line to initialize the InferenceTree object, and one to trigger the data processing.
```python
#As this model comes from huggingface, we define model_origin as "hf"
inference_tree_object = InferenceTree(model, model_origin="hf")
#data_type can either be praw or json, depending on whether the data is being passed directly as a praw object or loaded in via json.
output = inference_tree_object.process_thread(thread, data_type="json")
```
And you are done.  What this gives you is a list of strings, in this case being one string due to us using only default settings.  Print it or save it to a file, the summarized form of the conversation is complete.
The full example:
```python
#Import packages
import ConversationInferenceTree
import json
import os

#Import the sample data
with open("test_praw_data.json", "r", encoding="utf-8") as f:
    thread = json.load(f)

#Define the model repository ID
model_id = "meta-llama/Llama-3.1-8B-Instruct"

#Initialize the package object
inference_tree_object = InferenceTree(model_id, model_origin="hf")

#Trigger the processing, output is a list of strings
output = inference_tree_object.process_thread(thread, data_type="json")
```
## Customizing with params
### Setting up the model
As shown above in the minimum example, the simplest way to get a model working is to pass in its model repository ID, along with the model source.  As of the current build of ConversationInferenceTree, models can be sourced from two main places; huggingface and openai.
```python
inference_object = InferenceTree(model="meta-llama/Llama-3.1-8B-Instruct", model_origin="hf")
#OR
inference_object = InferenceTree(model="gpt-4o", model_origin="openai")
```
What is the difference between these two?  With huggingface, the model is internally loaded with AutoModelForCausalLM, and prompts are encoded with the model's tokenizer, generation is handled locally, then output is decoded and returned.  For OpenAI calls, prompts are passed through the OpenAI API to be handled remotely.  Please note that that this means that gpt API usage requires a valid openai key.

#### Prompt_type
prompt_type is a parameter that allows for explicit choice of how to pass a prompt to the model.  The two choices are: "question" and "role".
```python
inference_object = InferenceTree(model_name, "hf", prompt_type="role")
#This will format the internal prompts as:
prompt = [
    {"role": "system", "content": "<The agent question will go here>"},
    {"role": "user", "content": "<The text input will be put here>"},
]
```
```python
inference_object = InferenceTree(model_name, "hf", prompt_type="question")
#This will format the internal prompts as:
prompt = f"{<'The agent question will go here'>}\n{<'The text input will be put here'>}"
```

### Adding a summarizer
Summarizers are used to condense the output of agent processing across different depths of a Reddit comment thread. You can customize how the summarization is applied, how the prompt is structured, and how the result is formatted. You define summarizers by passing a list of dictionaries to the summarizer_list argument when creating your InferenceTree object.

Each summarizer must define:

- The query to be used for summarization
- The depth at which it operates
- An input_template to format the prompt(using python's format functionality)
- An output_template to format the result(using python's format functionality)
- input_vars and output_vars to customize the templates

There must be at least two summarizers:
- One at depth 0, though more can be defined, ascending through depth one at a time.
- At least one at depth: -1 for generating the final report

Example:
```python
summarizers = [
    {
        "query": "Summarize this comment section in simple terms.",
        "depth": 0,
        "input_template": "{text}",
        "input_vars": {},
        "output_template": "{prev_output}\n{gen}",
        "output_vars": {}
    },
    {
        "query": "Provide a final overview of the post and all its discussions.",
        "depth": -1,
        "input_template": "{intro}{root}{divider}{comment_summaries}",
        "input_vars": {
            "intro": "This is the main post content:\n",
            "divider": "\nHere is a summary of the conversation:\n"
        },
        "output_template": "{gen}",
        "output_vars": {}
    }
]

inference = InferenceTree(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_origin="hf",
    summarizer_list=summarizers
)
```
#### Mandatory Template Variables
Depending on the depth of the summarizer, certain variables must be present in the input_template:
- Depth 0 or higher: {text}

-- text -- the concatenated together totality of the agent outputs that this summarizer handles.
  
- Depth -1: {root} and {comment_summaries}

-- root -- Passes in the body text of the thread post body.
  
-- comment_summaries -- passes in the summarized content of all top-level variables.  

These are injected automatically by the system. Any additional placeholders must be accounted for in input_vars.



### Adding an agent
Agents are used to extract insights or perform custom processing on individual comments in the Reddit thread. Each agent is tied to a specific depth level and generates model outputs using a user-defined query and formatting templates. You define agents by passing a list of dictionaries to the question_list argument when creating your InferenceTree object.

Each agent must define:
- The query that will be asked during generation
- The depth at which it operates
- An input_template to format the model prompt (using Python's format functionality)
- An output_template to format the result (also using format)
- input_vars and output_vars to fill in any additional placeholders used in the templates
- At least one agent must be defined at depth 0. More agents can be added for higher depths (e.g., depth 1, depth 2, etc.).

Example:
```python
agents = [
    {
        "query": "What is the main idea of this comment?",
        "depth": 0,
        "input_template": "{text_body}{summary_header}{summary}",
        "input_vars": {
            "summary_header": "\nSummary of replies:\n"
        },
        "output_template": "{prev_output}\n\nQ: {query}\nA: {gen}",
        "output_vars": {}
    },
    {
        "query": "How does this reply build upon or contrast with the parent comment?",
        "depth": 1,
        "input_template": "{text_body}{summary}",
        "input_vars": {},
        "output_template": "{prev_output}\n\n[Question: {query}]\n[Response: {gen}]\n",
        "output_vars": {}
    }
]

inference = InferenceTree(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_origin="hf",
    question_list=agents
)
```
#### Mandatory Template Variables
Depending on the agent’s function, certain variables must be present in the input and output templates:

In the input_template:
- For all depths:

-- {text_body} — the raw content of the comment the agent is analyzing

-- {summary} — the summarized output from that comment's children

In the output_template:
- For all depths:

-- {prev_output} — accumulates output across multiple agents at the same depth

-- {query} — the agent’s question string (for context)

-- {gen} — the generated result returned by the model

These variables are also injected automatically by the system. If you include any custom placeholders (e.g., {summary_header}, {query_prefix}), they must be defined in input_vars or output_vars.

## License
[New license name here?](LICENSE)