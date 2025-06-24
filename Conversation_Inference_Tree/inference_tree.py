import praw
import os
from collections import deque, defaultdict

from .tree import _Tree
from .agent import _Agent
from .model_wrapper import _ModelWrapper
from .logger import logger

class InferenceTree:
    """
    The main class that is called by users using the package

    Methods:
        process_thread: takes the pre-loaded questions and llm and applies them to
        a reddit thread in var data
    Args:
        model_name: a string that the selected model handler uses to reference a specific llm
                    for example, a huggingface model could be "meta-llama/Llama-3.2-3B-Instruct"
        model_origin: selects a model handler.  If "hf" is passed in, then model_name model is a 
                      huggingface model.  If "openai" is passed in, then the questions are treated
                      as calls to the OpenAI API.
        question_list: a list of dictionaries containing the agent data.
        {
            "question" OR "role": put the question you want the agent to ask here as a string,
                                  choosing question causes the agent to format the prompt as one long
                                  string, while choosing role causes the agent to format the prompt as
                                  "user", "system", and so forth in a dictionary prompt.
            "depth": tells the agent where it needs to be applied.  0 are top level comments, 1 are replies
                     to those comments, 2 are replies to replies, and so forth.  -1 defines the summarizer.
            "order": NOT IMPLEMENTED YET
        }
        model_params: Use this dict variable to pass hyperparameters into the model, 
                      just as you normally would.
        children_per_summary: Defines how many child node outputs are concatenated together for each summary.

    """
    def __init__(self, model_name: str, model_origin: str, question_list: list[dict], model_params: dict = None, children_per_summary: int = 5,):
        self.agent_list = []
        self.children_per_summary = children_per_summary 

        self.llm = _ModelWrapper(model_name=model_name, model_origin=model_origin, model_params=model_params)
        
        #Sets the agents according to user specifications in question_list
        for question in question_list:
            logger.debug(f"setting agent for question: '{question["question"]}'")

            question_order = question.get('order', 1)

            self.agent_list.append(_Agent(
                                        query=question.get("question", "role"),#NOTE: Need to figure out role vs question handling 
                                        depth=question["depth"], 
                                        order=question_order
                                        )
                                    )
        logger.info("inference object initialized")

    def _do_summary_and_agent(self, tree):
        output_stack = deque()
        temp_holding = defaultdict(list)

        #Add top comment(post) to stack
        output_stack.append(tree.root)

        #Loop continues till the stack is completely emptied
        while output_stack:
            #current_holding contains the agent outputs of the children of the top-of-stack node
            #current_children contains the children of the top-of-stack node in a list
            current_holding = temp_holding.get(output_stack[-1], [])
            current_children = tree.children(output_stack[-1])

            #If the more child nodes than there are summaries of child nodes, add the next child to the stack
            if len(current_children) > len(current_holding):
                output_stack.append(current_children[len(current_holding)].identifier)
            else:
                logger.debug(f"processing for {output_stack[-1]}.  Children: {len(current_holding)}")
                
                summary = ""  #NOTE: Change this into a summary function for ease of reading?
                #if there is anything in current_holding, 
                if len(current_holding) > 0:
                    #Split the stored outputs into batches to be given to the summarizer
                    batch_holding = [current_holding[i:i + self.children_per_summary] for i in range(0, len(current_holding), self.children_per_summary)]
                    summarizer_agent = next((u for u in self.agent_list if u.depth == -1))
                    
                    #Summarize current_holding
                    for batch in batch_holding:
                        summary = f"{summary}{self.llm.generate(summarizer_agent.form_prompt("\n\n".join(batch)))}\n\n"
                    
                    #Clear the now-used entry in temp-holding
                    del temp_holding[output_stack[-1]]
                
                #if the current top-of-stack is root, return the result
                if output_stack[-1] == tree.root: return summary #NOTE: perhaps make this summary be separate so the user can define user readable formatting?

                #With context from summary, apply depth-appropriate agent(s) to top-of-stack node
                current_agents = [u for u in self.agent_list if u.depth == tree.get_node(output_stack[-1]).data.depth]
                current_agent_output = ""
                for a in current_agents:
                  #NOTE: Make the following customizable to the user, so they can define how the different components will be created.
                  current_agent_output = (f"{current_agent_output}The output for the question \"{a.query}\" is:\n{self.llm.generate(a.form_prompt(tree.get_node(output_stack[-1]).data.body))}\nHere is a summary of the response to this comment:\n{summary}\n\n")
                  
                #Remove completed node from top of the stack
                output_stack.pop()
                
                #append output from agents to the temp_holding level keyed to the new top-of-stack
                temp_holding[output_stack[-1]].append(current_agent_output)
        #If the while loop completes, then the return statement never triggered
        raise Exception("Return statement never triggered, likely a problem with tree initialization!")
    #process reddit thread NOTE: input_location and output_location not implemented yet
    def process_thread(self, data, data_type: str, output_location: str = None):

        #If input_location is not equal to "", pull data from json files
        conversation_tree = _Tree(data).tree
        logger.info("tree populated")
        
        inference_summary = self._do_summary_and_agent(conversation_tree)

        logger.info("All conversations processed")
        
        if output_location is not None:
            text_file = open(output_location, "w")
            text_file.write(inference_summary)
            text_file.close()
        else: return inference_summary
    