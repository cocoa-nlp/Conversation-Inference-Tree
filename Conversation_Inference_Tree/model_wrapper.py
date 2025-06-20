from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
import openai

from .logger import logger

class _ModelWrapper:
    def __init__(self, model_name: str, model_origin: str, model_params: dict = None):
        self.model_origin = model_origin     

        load_dotenv()
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
        
        #sets the llm that will be used by the other functions, and exposes it as an accessible variable
        #NOTE: Check out python logging package
        if model_origin == "hf":
            #This code runs if the llm is from huggingface.co or a local huggingface model
            key = os.getenv('token')
            # user = os.getenv('username')

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
    
    def generate(self, input):
        if self.model_origin == "hf":
            response = self.model(input)
            return response
        elif self.model_origin == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model["model"],
                    message=input,
                    config=self.model["config"]
                )
                return response['choices'][0]['message']['content']
        else:
            logger.error("model failed generation step")