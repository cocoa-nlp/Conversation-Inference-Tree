from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
import openai
from treelib import Node, Tree
import praw

class InferenceTree:
    model = None
    agent_dict = dict()
    agent_output_dict = dict()
    max_summary_nodes = 5
    load_dotenv()

    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def set_summarizer(instruction: str, max_nodes=5):
        agent = {
            "query": instruction
        }
        #adds the summarizer to the agent dictionary
        InferenceTree.agent_dict["_summarizer"] = agent

    def set_llm(model_type: str, model_name: str, model_parameters: dict):
        #sets the llm that will be used by the other functions, and exposes it as an accessible variable
        if model_type == "huggingface":
            key = os.getenv('token')
            # user = os.getenv('username')

            #save the user-defined model hyperparameters to an AutoConfig object, then initialize the model into a global variable
            config = AutoConfig.from_pretrained(model_name, **model_parameters)
            automodel = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            autotokenizer = AutoTokenizer.from_pretrained(model_name)
            InferenceTree.model = pipeline("text-generation", model=automodel, tokenizer=autotokenizer)
        elif model_type == "openai":
            key = os.getenv('OPENAI_API_KEY')
            InferenceTree.model = {
                "model": model_name,
                "config": model_parameters
            }
    #process reddit thread
    def process_thread(data, data_type: str, input_location: str, output_location: str):
        pass

class _Agent:
    def _general_agent(model, prompt): #--Handles the basic processing of all agents
        try:
            if isinstance(model, dict):
                response = openai.ChatCompletion.create(
                    model=model["model"],
                    message=prompt,
                    config=model["config"]
                )
                return response['choices'][0]['message']['content']
            else:
                response = model(prompt)
                return response
        except Exception as e:
            print(f"Error generating agent output: {e}")
    
        #set node agent
    def set_node_agent(agent_name: str, query: str, input: str):
        agent = {
            "query": query,
            "input": input
        }
        #adds the agent to the agent dictionary
        InferenceTree.agent_dict[agent_name] = agent

class _ConversationTree:
    def __init__(self, submission):
        self.tree = Tree()

        submission.comments.replace_more(limit=None)

        # Add root node (submission itself)
        self.tree.create_node("root-node", submission.id, data={
            #NOTE: fill out dictionary
        })

        # Start recursion from top-level comments
        for top_level_comment in submission.comments:
            self._recursive_node(top_level_comment, submission.id)
    
    #sets the subcomments of entry as child nodes, and repeats the chain
    #does not handle setting the entry node itself, as that would make setting the root complicated
    def _recursive_node(self, entry):
        #entry will be a comment object with 0 or more subcomments

        if len(entry.replies) != 0:
            for child in entry.replies:
                self.tree.create_node(child.body[:30], child.id, parent=entry.id, data={
                    #NOTE: fill out dictionary
                })
                
                self.tree = self._recursive_node(child)
        

    def set_tree(input, inputtype):
        #options for inputtype: link, praw, psaw(for future)

        if inputtype == "link":
            #NOTE: needs secret and id
            reddit = praw.Reddit(client_id='id', client_secret='secret', user_agent='ua')
            submission = reddit.submission(url=input)

            #allows for full expansion of threads
            submission.comments.replace_more(limit=0)
            input = submission
        
        #translate other types of input into praw object

        #set root node

        #for each comment in submission