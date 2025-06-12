import praw
import os

from .tree import _Tree
from .agent import _Agent
from .model_wrapper import _ModelWrapper

class InferenceTree:
    def __init__(self, model_name: str, model_type: str, question_list: list[dict], model_params: dict = None, max_per_summary: int = 5,):
        self.agent_list = []
        self.max_per_summary = max_per_summary 

        self.llm = _ModelWrapper(model_name=model_name, model_type=model_type, model_params=model_params)
        
        #Sets the agents according to user specifications in question_list
        for question in question_list:
            print(f"setting agent for question: \"{question["question"]}\"")
            try:
                question_order = question["order"]
            except KeyError:
                question_order = 1

            self.agent_list.append(_Agent(
                                        query=question["question"], 
                                        depth=question["depth"], 
                                        order=question_order
                                        )
                                    )

    #process reddit thread NOTE: input_location and output_location not implemented yet
    def process_thread(self, data, data_type: str, output_location: str = None):

        #If input_location is not equal to "", pull data from json files
        conversation_tree = _Tree(data)
        
        print("Tree initialization complete")
        exit()
        if output_location is not None:
            text_file = open(output_location, "w")
            text_file.write(inference_summary)
            text_file.close()