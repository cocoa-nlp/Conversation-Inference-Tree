import praw
import os
from collections import deque

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

    #process reddit thread NOTE: input_location and output_location not implemented yet
    def process_thread(self, data, data_type: str, output_location: str = None):

        #If input_location is not equal to "", pull data from json files
        conversation_tree = _Tree(data).tree
        logger.info("tree populated")
        
        #Pseudocode start
        
        output_stack = deque()
        temp_holding = dict()

        #Add top comment(post) to stack
        output_stack.append(conversation_tree.root)

        #Loop continues till the stack is completely emptied
        while output_stack:
            #current_holding contains the agent outputs of the children of the top-of-stack node
            #current_children contains the children of the top-of-stack node in a list
            current_holding = temp_holding.get(output_stack[-1], [])
            conversation_tree.show()
            current_children = conversation_tree.children(output_stack[-1])

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
                    for batch in batch_holding: #NOTE: add options for summarization that aren't exclusively ordered batches
                        summary = summary + self.llm.generate(summarizer_agent.form_prompt("\n\n".join(batch))) + "\n\n"
                    
                    #Clear the now-used entry in temp-holding
                    del temp_holding[output_stack[-1]]
                
                #With context from summary, apply depth-appropriate agent(s) to top-of-stack node
                current_agents = [u for u in self.agent_list if u.depth == conversation_tree.get_node(output_stack[-1]).data.depth]
                current_agent_output = ""
                for a in current_agents:
                    #NOTE: Make the following customizable to the user, so they can define how the different components will be created.
                    current_agent_output = (current_agent_output #If there are multiple agent questions to be asked, they will concatenate
                                            + "The output for the question: '"+ a.query + "' is:\n" 
                                            + self.llm.generate( #Get a response from the llm for the following
                                                a.form_prompt(#Format a prompt from the following input text
                                                    conversation_tree.get_node(output_stack[-1]).data.body)
                                                    + "\nHere is a summary of the response to this comment:\n" #Add summary for context
                                                    + summary)#NOTE:^ I should move this to an earlier if statment so that if there is no comment due to 0 replies to current node, there aren't unnessessary tokens added to prompt
                                            + "\n\n") #Add a little space between agent responses in output string
                
                #Remove completed node from top of the stack
                output_stack.pop()
                
                #append output from agents to the temp_holding level keyed to the new top-of-stack
                temp_holding[output_stack[-1]].append(current_agent_output)
        
        logger.info("All conversations processed")
        exit()
        if output_location is not None:
            text_file = open(output_location, "w")
            text_file.write(inference_summary)
            text_file.close()
        else: return inference_summary
    