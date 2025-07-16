import praw
import os
from collections import deque, defaultdict

from .tree import _Tree
from .agent import _Agent
from .model_wrapper import _ModelWrapper
from .logger import logger
from .CLIGraph import CLIGraph

class OutputFormatter:
    def __init__(self, template: str, user_vars: dict):
        self.template = template
        self.user_vars = user_vars
    
    def format(self, vars: dict):
        try:
            return self.template.format(**self.user_vars, **vars)
        except KeyError as e: #TODO: add logging functionality.
            raise ValueError(f"Missing placeholder in template: {e}")

class InferenceTree:
    """
    The main class that is called by users using the package

    Methods:
    process_thread(data, data_type, output_location) -- 
        takes the pre-loaded questions and llm and applies them to a reddit thread in var data
    _do_summary_and_agent(tree) -- 
        a private function that holds the processing logic of applying the
        various agents, for the purposes of adding abstraction for greater
        readability.  Takes the tree object containing all conversation data.
    
    Args:
    model_name -- a string that the selected model handler uses to reference a specific llm
                    for example, a huggingface model could be "meta-llama/Llama-3.2-3B-Instruct"
    model_origin -- selects a model handler.  If "hf" is passed in, then model_name model is a 
                    huggingface model.  If "openai" is passed in, then the questions are treated
                    as calls to the OpenAI API.
    summarizer_list -- a list of dictionaries containing one or more summary questions to be applied to 
                  child outputs.  Same args as question_list below
    question_list -- a list of dictionaries containing the agent data.
        {
            "question" OR "role": put the question you want the agent to ask here as a string,
                                  choosing question causes the agent to format the prompt as one long
                                  string, while choosing role causes the agent to format the prompt as
                                  "user", "system", and so forth in a dictionary prompt.
            "depth": tells the agent where it needs to be applied.  0 are top level comments, 1 are replies
                     to those comments, 2 are replies to replies, and so forth.
            "order": NOT IMPLEMENTED YET
        }
    model_params -- Use this dict variable to pass hyperparameters into the model, 
                    just as you normally would.
    prompt_type -- default is "role".  Defines how the _Agent object formats the input to the llm.  Options are "question" and "role"
    children_per_summary -- Defines how many child node outputs are concatenated together for each summary.
    graph -- default True: a boolean value that determines whether a graph displaying a live view of the current processing node depth is displayed.
    """ 
    def __init__(
        self, 
        model_name: str, 
        model_origin: str, 
        summarizer_list: list[dict] = [
            {
                "question": "Summarize the text in 150 words or less.",
                "depth": -99 #NOTE: is -99 for now to avoid keyerrors when wrapping into agent
            },
            {
                "question": "Give a report on the reddit post, along with its following text bodies containing information about the conversations it started.",
                "depth": -99 #NOTE: is -99 for now to avoid keyerrors when wrapping into agent                
            }
        ],
        question_list: list[dict] = [
            {
                "question": "Tell me what the subject of this conversation is, and the sentiment expressed about the subject.",
                "depth": 0,
            },
            {
                "question": "Tell me what this reply is talking about and what the author probably feels about the subject.",
                "depth": 1
            },
        ], 
        model_params: dict = {}, 
        agent_format: OutputFormatter = OutputFormatter(
            template = "{prev_output}{question_prefix}{question}{question_suffix}{gen}{sep}",
            user_vars = {
                "question_prefix": "The output for the question \"",
                "question_suffix": "\" is:\n",
                "sep": "\n\n"
            }
        ),
        agent_input_format: OutputFormatter = OutputFormatter(
            template = "{text_body}{summary_prefix}{summary}",
            user_vars = {
                "summary_prefix": "\nHere is a summary of the response to this comment:\n"
            }
        ),
        summary_format: OutputFormatter = OutputFormatter(
            template = "{prev_output}{gen}{sep}",
            user_vars = {"sep": "\n\n"}
        ), 
        final_summary_input_format: OutputFormatter = OutputFormatter(
            template = "{root_prefix}{root}{sep}{comment_summaries}",
            user_vars = {
                "root_prefix": "Here is the body text of the post:\n",
                "sep": "\nHere is a number of summaries of the post comments' content:\n"
            }
        ),
        prompt_type: str = "role",
        children_per_summary: int = 5,
        graph: bool = True
    ):
        self.agent_list = []
        self.children_per_summary = children_per_summary 
        self.agent_format = agent_format
        self.agent_input_format = agent_input_format
        self.summary_format = summary_format
        self.final_summary_input_format = final_summary_input_format
        self.graph = graph

        self.llm = _ModelWrapper(model_name=model_name, model_origin=model_origin, model_params=model_params)
        
        #Convert summarizer_list into wrapped agents
        self.summarizer_list = []
        for summarizer in summarizer_list:
            self.summarizer_list.append(_Agent(
                                            query=summarizer.get("question"),
                                            depth=summarizer["depth"], 
                                            prompt_type=prompt_type
                                        ))

        #Sets the agents according to user specifications in question_list
        for question in question_list:
            logger.debug(f"setting agent for question: '{question['question']}'")

            question_order = question.get('order', 1)
            if question["depth"] < 0 and question["depth"] != -99:
                raise ValueError(f"The question {question} was set to an incorrect value")
            self.agent_list.append(_Agent(
                                        query=question.get("question"), 
                                        depth=question["depth"], 
                                        order=question_order,
                                        prompt_type=prompt_type
                                        )
                                    )
        logger.info("inference object initialized")
    
    def _split_into_batches(self, target: list, batch_size: int):
        """Takes a list, and splits into a list of sublists of the origional list each with a size of batch_size"""
        batches = [target[i:i + batch_size] for i in range(0, len(target), batch_size)]
        return batches
    
    def _get_agents(self, tree_object, top_stack_id):
        """returns the agents for the depth of the current top-of-stack node in a list"""
        #Pull out the top-of-stack node object from the treelib
        current_depth_agents = []
        current_depth = tree_object.get_node(top_stack_id).data.depth
        while len(current_depth_agents) == 0:
            #Use list comprehension to store all agents at the correct depth in a list
            current_depth_agents = [a for a in self.agent_list if a.depth == current_depth]
            current_depth -= 1
        return current_depth_agents
    
    def _do_agent_processing(self, current_agents, tree_object, top_stack_id, prev_summary):
        """
        This function gets the generated llm output from the current-depth agents together
        with the current node's text body.

        Args:
        current_agents -- the list of agents that apply at the current node's depth
        tree-object -- the tree holding all nodes
        top_stack_id -- the id of the current top-of-stack node
        prev_summary -- a string containing the summarized output from children of the current top-of-stack

        Returns:
        output -- A string containing all agent outputs, formatted and concatenated together with an fstring
        """
        #Pull out the top-of-stack node object from the treelib
        top_stack_node = tree_object.get_node(top_stack_id)
        output = ""
        for a in current_agents:
            text = self.agent_input_format.format({
                "text_body": top_stack_node.data.body,
                "summary": prev_summary
            })
            if text == "":
                print()
                print("Agent gen call -- Text failed to format correctly and ended up as an empty string!")
                print(f"- top_stack_node.data.body = {top_stack_node.data.body}")
                print(f"- prev_summary = [prev_summary]")
                exit()
            output = self.agent_format.format({
                "prev_output": output,
                "question": a.query,
                "gen": self.llm.generate(text, a),
            })
            if output == "":
                print("Output returned an empty string!")
        return output

    def _do_summary_processing(self, current_holding: list):
        """
        This function takes the agent output from a node's child nodes and uses the llm to get a summary.

        Args:
        current_holding -- A list of a given node's child node agent outputs in single string form

        Returns:
        output -- A single string containing the summary
        """
        #if there is anything in current_holding, 
        if len(current_holding) > 0:
            #Split the stored outputs into batches to be given to the summarizer
            batch_holding = self._split_into_batches(current_holding, self.children_per_summary)
            
            #Summarize current_holding
            output = ""
            for batch in batch_holding:
                batch_text = "\nNext Summary Text:\n".join(batch)

                output = self.summary_format.format({
                    "prev_output": output,
                    "gen": self.llm.generate(batch_text, self.summarizer_list[0])
                })
            return output

    def _do_summary_and_agent(self, tree):
        """
        This function contains the main logic of the script, and handles the loop that traverses through each comment
        tree until the the stack "output_stack" has been depleted down to the root-node.

        Args:

        Returns:

        """
        output_stack = deque()
        temp_holding = defaultdict(list)

        #Add top comment(post) to stack
        output_stack.append(tree.root)

        #Loop continues till the stack is completely emptied
        #   It doesn't sit right to have a while loop where termination is a failure condition
        g = CLIGraph(-1, 7, desc='Processing at depth')
        while output_stack:
            #current_holding contains the agent outputs of the children of the top-of-stack node
            #current_children contains the children of the top-of-stack node in a list
            current_holding = temp_holding.get(output_stack[-1], []) #TODO: Change to current_children_outputs
            current_children = tree.children(output_stack[-1])

            #If the more child nodes than there are summaries of child nodes, add the next child to the stack
            if len(current_children) > len(current_holding):
                output_stack.append(current_children[len(current_holding)].identifier)
            else:
                logger.debug(f"processing for {output_stack[-1]}.  Children: {len(current_holding)}")
                if self.graph: g.update(tree.get_node(output_stack[-1]).data.depth)

                summary = self._do_summary_processing(current_holding)

                #if the current top-of-stack is root, return the result
                if output_stack[-1] == tree.root:
                    formatted_output = self.final_summary_input_format({
                        "root": tree.get_node(output_stack[-1]).data.body,
                        "comment_summaries": summary
                    })
                    return self.llm.generate(formatted_output, self.summarizer_list[1])

                #Clear the now-used entry in temp-holding
                try:
                    del temp_holding[output_stack[-1]]
                except KeyError:
                    logger.debug(f"Node {output_stack[-1]} has no children, skipping tempholding slot clearing")

                #With context from summary, apply depth-appropriate agent(s) to top-of-stack node
                current_agents = self._get_agents(tree, output_stack[-1])
                current_agent_output = self._do_agent_processing(current_agents, tree, output_stack[-1], summary)             
                
                #append output from agents to the temp_holding level keyed to the new top-of-stack
                temp_holding[tree.get_node(output_stack[-1]).data.parent_id].append(current_agent_output)

                if current_agent_output == "":
                    print(f"current_agent_output is an empty string for node {output_stack[-1]} and was added as a child of node {tree.get_node(output_stack[-1]).data.parent_id}")   

                #Remove completed node from top of the stack
                output_stack.pop()

        #If the while loop completes, then the return statement never triggered
        raise Exception("Return statement never triggered, likely a problem with tree initialization!")

    def process_thread(self, data, data_type: str):
        """
        This high-level function triggers the inference of a passed in conversation, by loading the conversation into
        a treelib structure then passing data to _do_summary_and_agent for logic handling
        
        Args:
        data -- the conversation thread.  No datatype is defined because unification of datatypes is handled lower in the logic.
        data_type -- a string that defines what format the data was passed in.  Can be "json", "praw", or "psaw"

        Returns:
        inference_summary -- A string containing the final summary of the input thread
        """
        #If input_location is not equal to "", pull data from json files
        conversation_tree = _Tree(data).tree
        logger.info("tree populated")

        inference_summary = self._do_summary_and_agent(conversation_tree)

        logger.info("All conversations processed")
        
        return inference_summary
    