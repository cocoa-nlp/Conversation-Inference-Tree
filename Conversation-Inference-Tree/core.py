from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
import openai

class InferenceTree:
    #set summarizer
    load_dotenv()
    # user = os.getenv('username')

    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def set_summarizer(instruction: str, max_nodes=5):
        #this is the agent that will summarize the agent commentary between each time step
        #will pass the text input as a list of strings and the instruction string to the general agent function
        pass

    #set node agent
    def set_node_agent(agent_name: str, query: str, text_input: str, target):
        #this is the agent that will be used to process the nodes in the comment tree
        #will pass the text input as a list of strings and the instruction string to the general agent function
        #will also pass the node name and node query string to the general agent function
        #NOTE: need to know how to define where the node will be applied in the comment tree
        pass

    def set_llm(model_type: str, model_name: str, model_parameters: dict):
        global model
        #sets the llm that will be used by the other functions, and exposes it as an accessible variable
        if model_type == "huggingface":
            key = os.getenv('token')

            #save the user-defined model hyperparameters to an AutoConfig object, then initialize the model into a global variable
            config = AutoConfig.from_pretrained(model_name, **model_parameters)
            automodel = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            autotokenizer = AutoTokenizer.from_pretrained(model_name)
            model = pipeline("text-generation", model=automodel, tokenizer=autotokenizer)
        elif model_type == "openai":
            key = os.getenv('OPENAI_API_KEY')
            model = {
                "model": model_name,
                "config": model_parameters
            }


        #constructor without the key
    
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
                response = model.generate(
                    input_ids=prompt
                )
                return response
        except Exception as e:
            print(f"Error generating agent output: {e}")

    #process reddit thread
    def process_thread(data, data_type: str, input_location: str, output_location: str):
        pass
