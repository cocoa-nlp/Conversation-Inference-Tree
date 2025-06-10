from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import praw
import os

from .tree import _Tree
from .agent import _Agent

#Todo List:
    #Add the code in the process function that brings together the query and the input
    #Figure out where the text input is coming from in this case
    #
    #Change the agent storage so that it accepts Agent object instead of sub-dictionaries
    #
    #Finish ProcessThread function in InferenceTree class

class InferenceTree:
    llm = None
    agent_list = []
    max_summary_nodes = 5
    load_dotenv()

    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def set_agent(instruction: str, depth: int, max_nodes=5): #NOTE: remove maxnodes?
        agent = _Agent(instruction, depth)
        InferenceTree.agent_list.append(agent)

#NOTE: bring outside of InferenceTree?
    def set_llm(model_name: str, model_type: str, model_parameters: dict):
        #sets the llm that will be used by the other functions, and exposes it as an accessible variable
        #NOTE: Check out python logging package
        if model_type == "huggingface":
            #This code runs if the llm is from huggingface.co or a local huggingface model
            key = os.getenv('token')
            # user = os.getenv('username')

            #save the user-defined model hyperparameters to an AutoConfig object, then initialize the model into a global variable
            config = AutoConfig.from_pretrained(model_name, **model_parameters)
            automodel = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            autotokenizer = AutoTokenizer.from_pretrained(model_name)
            #NOTE: Change to instance-level variable
            InferenceTree.llm = pipeline("text-generation", model=automodel, tokenizer=autotokenizer)
        elif model_type == "openai":
            #This code runs if the llm is accessed throught the openai api
            key = os.getenv('OPENAI_API_KEY')
            InferenceTree.llm = {
                "model": model_name,
                "config": model_parameters
            }
    #process reddit thread NOTE: input_location and output_location not implemented yet
    def process_thread(self, data, data_type: str, input_location: str = "", output_location: str = ""):
        #If input_location is not equal to "", pull data from json files
        thread = _Tree(data)

        print("Tree initialization complete")
        exit()