from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
import openai

class _ModelWrapper:
    def __init__(self, model_name: str, model_type: str, model_params: dict = None):
        self.model_type = model_type     

        load_dotenv()
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
        
        #sets the llm that will be used by the other functions, and exposes it as an accessible variable
        #NOTE: Check out python logging package
        if model_type == "hf":
            #This code runs if the llm is from huggingface.co or a local huggingface model
            key = os.getenv('token')
            # user = os.getenv('username')

            #save the user-defined model hyperparameters to an AutoConfig object, then initialize the model into a global variable
            config = AutoConfig.from_pretrained(model_name, **model_params)
            automodel = AutoModelForCausalLM.from_pretrained(model_name, config=config)
            autotokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = pipeline("text-generation", model=automodel, tokenizer=autotokenizer)
        elif model_type == "openai":
            #This code runs if the llm is accessed throught the openai api
            key = os.getenv('OPENAI_API_KEY')
            self.model = {
                "model": model_name,
                "config": model_params
            }
        else: print("Model type selected incorrectly")
    
    def generate(self, input):
        try:
            if self.model_type == "hf":
                response = self.model(input)
                return response
            elif self.model_type == "openai":
                    response = openai.ChatCompletion.create(
                        model=self.model["model"],
                        message=input,
                        config=self.model["config"]
                    )
                    return response['choices'][0]['message']['content']
            else: print("Model type selected incorrectly")
        except Exception as e:
            print(f"Error generating agent output: {e}")  