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
        conversation_tree = _Tree(data).tree
        
        #Starting at the deepest horizontal depth in the tree iterate through each depth up to root
        output_list = []
        for d in range(conversation_tree.depth(), -1, -1):
            #get a list of all nodes at current depth
            d_nodes = [node for node in conversation_tree.all_nodes() if conversation_tree.depth(node.identifier) == d]

            for d_node in d_nodes:
                #if the node has any children
                if len(conversation_tree.children(d_node.identifier)) > 0:
                    #summarize all child nodes of current node
                    current_nodes_child_ids = [child.identifier for child in conversation_tree.children(d_node.identifier)]
                    #NOTE: UNFINISHED
                    #Pseudocode block start
                    #Pull the "content" dict entries from output_list that have dict id entries that match one string entry from current_nodes_child_ids
                    #Do summarizer, store in string variable children_summary
                    #Pseudocode block end

                    #with context output of summarizer, apply current depth agents
                    current_depth_agents = []
                    d_agent_counter = d
                    while len(current_depth_agents) == 0:
                        current_depth_agents = [a for a in self.agent_list if a.depth == d_agent_counter]
                        d_agent_counter -= 1
                    
                    for current_depth_agent in current_depth_agents:
                        output_list.append(
                            {
                                "id": d_node.identifier,
                                "content": self.llm.generate(current_depth_agent.form_prompt(children_summary))
                            }
                        )
                    

        print("Tree initialization complete")
        exit()
        if output_location is not None:
            text_file = open(output_location, "w")
            text_file.write(inference_summary)
            text_file.close()
    