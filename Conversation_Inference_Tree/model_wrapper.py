from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
from huggingface_hub import login
import openai

from .logger import logger
from .agent import _Agent

class _ModelWrapper:
    """
    This is a class used to contain and abstract the usage of a given llm for inference_tree.
    With this class, a user can pass agent prompts to either a local huggingface model OR the
    OpenAI API, without any additional tweaks outside of initialization.

    Methods:
        generate: takes the arguement "input", and passes it to the model for processing before
                  returning the model output.  Make sure to use the agent class's form_prompt()
                  function on an input to get it in the right format before passing.
    
    Args:
        model_name: The reference name for the model.  Example: "meta-llama/Llama-3.2-3B-Instruct" for
                    huggingface, or "gpt-4o" for an OpenAI API call.
        model_origin: Defines whether the model in question comes from huggingface(pass in "hf") or 
                      OpenAI(pass in "openai").
        model_params: Takes a dictionary of hyperparameters and directly applies them to the model.
                      Design this dictionary the same way as if you were calling the given llm directly.
    """

    def __init__(self, model_name: str, model_origin: str, model_params: dict = None):
        self.model_origin = model_origin     

        load_dotenv()
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
        
        #sets the llm that will be used by the other functions, and exposes it as an accessible variable
        if model_origin == "hf":
            #This code runs if the llm is from huggingface.co or a local huggingface model
            try:
                key = os.getenv('token')
                # user = os.getenv('username')
                login(key)
            except:
                raise KeyError("Login with token failed!  Make sure to set your huggingface login key under 'token' in your .env!")

            #save the user-defined model hyperparameters to an AutoConfig object, then initialize the model into a global variable
            config = AutoConfig.from_pretrained(model_name, **model_params)
            automodel = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            autotokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = pipeline("text-generation", model=automodel, tokenizer=autotokenizer)
        elif model_origin == "openai":
            #This code runs if the llm is accessed throught the openai api
            key = os.getenv('OPENAI_API_KEY')
            self.model = {
                "model": model_name,
                "config": model_params
            }
        else: 
            logger.error("model origin selected incorrectly, failed to load model")
            #raise TypeError(f"model_origin of '{model_origin}' incorrect, must be 'hf', 'openai'")
    
    def generate(self, input: str, agent: _Agent):
        formatted_input = agent.form_prompt(input)

        if self.model_origin == "hf":
            response = self.model(formatted_input)
            logger.debug(f"openai call for prompt\n{formatted_input}\ngave output\n{response}")
            return response
        elif self.model_origin == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model["model"],
                    message=formatted_input,
                    config=self.model["config"]
                )
                logger.debug(f"openai call for prompt\n{formatted_input}\ngave output\n{response['choices'][0]['message']['content']}")
                return response['choices'][0]['message']['content']
        else:
            logger.error("model failed generation step")