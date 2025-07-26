import praw
import os
import json
from collections import deque, defaultdict

from .tree import _Tree
from .agent import _Agent
from .model_wrapper import _ModelWrapper
from .logger import logger
from cligraph import CLIGraph



class InferenceTree:
    """
    The main class that is called by users using the package

    Methods:
    process_thread(data, data_type) -- takes the pre-loaded questions and llm and applies them to a reddit thread 
                                       in var data.  The only public-facing function.
    _do_summary_and_agent(tree) -- Holds the processing logic of applying the various agents, for the purposes of 
                                   adding abstraction for greater readability. Takes the tree object containing all conversation data.
    _split_into_batches(target, batch_size) -- Splits list "target" into a list of lists of size "batch_size" for batching purposes.
    _get_agents(tree_object, top_stack_id) -- Handles agent retrieval from the agent_list variable, filtering by the depth of the
                                              comment currently being processed.
    _do_agent_processing(
        current_agents, 
        tree_object, 
        top_stack_id, 
        prev_summary) -- Handles the generation step, using the content of the current question and the content of the
                         output from _get_agents.
    _do_summary_processing(current_holding) -- Similar functionality to _do_agent_processing, but customized for the summary step.

    Args:
    model_name -- a string that the selected model handler uses to reference a specific llm
                    for example, a huggingface model could be "meta-llama/Llama-3.2-3B-Instruct"
    model_origin -- selects a model handler.  If "hf" is passed in, then model_name model is a 
                    huggingface model.  If "openai" is passed in, then the questions are treated
                    as calls to the OpenAI API.
    summarizer_list -- A list of dictionaries defining the behavior and structure of the summarizing agents.  When manually initialized,
                       the user may provide two or more summarizers, one having a depth of -1 and one of 0.  There are two main pieces defined
                       within the dictionary; the actual agent, and a pair of formatters that can optionally be set by the user to have more
                       fine-tuned control over prompt process.  Below are the keys that must be defined in order to add a summarizer:
    {
        "query": A string containing the query that will be asked upon the calling of this _Agent object.  Depending on what prompt-type 
                 the user defines, this should be adjusted to fit it.  For "role", this should be an instruction sentence, such as "Summarize
                 this content in 100 words".  For "question", this would be concatenated together with text content to make one large string
                 to be passed to the model.
        "depth": An int.  Depth for summarizers is different from the agents of question_list.  There must be at least one summarizer with a depth of -1 
                 and one with 0.  -1 depth defines the summarizer that will create the final report that is passed back as the final product, 
                 and the user may define as many as they wish, with a discrete report being generated and appended to the return list for each 
                 summarizer respectively.  For summarizers 0 and up, the user may define one of each, with each one being applied to aggregate 
                 the data for agent outputs at its same depth.  For example, top-level-comments have a depth of 0, so the summarizer with a depth
                 of 0 will be applied to them.  Depth 1 would be applied to the replies to that comment, so forth and so on.  If a set of comments
                 at a given depth must be summarized together but do not have a summmarizer at that depth, the package will iterate towards 0 one
                 depth at a time until a summarizer is found, meaning that the summarizer with the greatest depth will act as a catchall.
        "input_template": This uses the formatting function provided by python.  This should be a string that looks like this: "{var1}{var2}{etc...}"
                          Any number of variables may be defined in the template, but there are mandatory variables that must be added that allow the processed text
                          to be added in.  
                          For summarizers with a depth of -1, the user MUST include {root} and {comment_summaries}.  
                              root -- Passes in the body text of the thread post body.
                              comment_summaries -- passes in the summarized content of all top-level variables.  
                          For all summarizers with a depth of 0 and up, the mandatory variable is {text}.
                              text: the concatenated together totality of the agent outputs that this summarizer handles.
        "input_vars": A dictionary of strings.  If the user adds any non-mandatory variables in the template, those variables MUST be represented as a key in
                      this dictionary.  For example, if we take the template "{var1}{text}{var2}", and this is the input template for a summarizer of depth 0, 
                      "text would be the mandatory variable, so var1 and var2 must both be represented like the following:
                      {
                        "var1": "User defined text..."
                        "var2": "more user text..."
                      }
        "output_template": A dictionary in the same form as input_template, but allows the user to define how the string will be saved as just AFTER llm generation
                           mandatory variables for summarizers with depth -1 are {prev_output} and {gen}.  Good for defining how batched summary rounds are concatenated
                           together.
                               prev_output -- A string that starts as "".  Holds the aggregated content from previous batches sent to the summarizer.  If this batch is not
                                              the last one, the product of this formatting will be part of the prev_output for the next output_template round.
                               gen -- The direct string output from the llm generation step. #NOTE: needs definition for other summarizer(just gen for mandatory)
        "output_vars": Same rules as input vars.  Account for all user-defined variables here 
    }
    question_list -- a dictionary similar to summarizer list.  However, this defines the agents that will apply questions to the reddit thread at given depths.
                     Must define at least one agent at depth 0(or more than one), and then may define as many agents per depth as wished.  If more than one agent is
                     defined for a depth, the agent step will loop through once per question, with the all outputs being aggregated together into a string at the end.
    {
        "query": A string containing the query that will be asked upon the calling of this _Agent object.  Depending on what prompt-type 
                the user defines, this should be adjusted to fit it.  For "role", this should be an instruction sentence, such as "Summarize
                this content in 100 words".  For "question", this would be concatenated together with text content to make one large string
                to be passed to the model.
        "depth": An int.  Defines at what depth of the conversation this agent's question will be used.  Agents with a depth of 0 will be applied to top-level-comments, 
                depth 1 will be used on replies, depth 2 on replies of replies, etc...
        "input_template": Same rules as input_template for summarizer's input_template.  Mandatory variables are {text_body} and {summary}.
            text_body -- the text body content of the comment the agent is being applied to.
            summary -- the summarized content of the reply's children.
        "input_vars": Same rules as summarizer's input_vars
        "output_template": Same rules as summarizer's output_template.  Mandatory variables are {prev_output}, {query}, and {gen}
                        prev_output -- A string that starts as "".  Holds the aggregated content all current_depth agents.  If this agent is not
                                        the last one, the product of this formatting will be part of the prev_output for the next output_template round.
                        query -- The question that the agent asked during the generation step.  Allows for context to be added to output.
                        gen -- The direct string output from the llm generation step.
        "output_vars"  Same rules as summarizer's output_vars
    }
    prompt_type -- "question" or "role".  Defines whether how the llm will recieve the prompt.  "role" uses the "user", "content", "system", "content" dictionary style.
                   "question" uses the large string method, with the question and the text input being appended together and passed in raw.  Should be set in accordance with
                   the model chosen and the agents defined.
    children_per_summary -- defines how many child agent outputs will be aggregated into a batch per summarizer round.  Default 5, larger values provide greater context but 
                            may overwhelm model, smaller values are better for smaller models.
    model_params -- A dictionary containing all hyperparameters the user wishes to use with the model.  Define in the same way one would do if working with the model directly.
    graph -- default True: a boolean value that determines whether a graph displaying a live view of the current processing node depth is displayed.  It is recommended to turn
                           this value to false when added into a fully automated script.
    """ 
    def __init__(
        self, 
        model_name: str, 
        model_origin: str, 
        summarizer_list: list[dict] = [
            {
                "query": "Summarize the text in 150 words or less.",
                "depth": 0,
                "input_template": "{text}",
                "input_vars": {},
                "output_template": "{prev_output}{sep}{gen}",
                "output_vars": {
                    "sep": "Next summary text:"
                }

            },
            {
                "query": "Give a thorough report on the reddit post, along with its following text bodies containing information about the conversations it started.",
                "depth": -1,
                "input_template": "{prefix}{root}{sep}{comment_summaries}",
                "input_vars": {
                    "prefix": "Here is the body text of the post:\n",
                    "sep": "\nHere is a number of summaries of the post comments' content:\n"
                },
                "output_template": "{gen}",
                "output_vars": {}
            }
        ],
        question_list: list[dict] = [
            {
                "query": "Tell me what the subject of this conversation is, and the sentiment expressed about the subject.",
                "depth": 0,
                "input_template": "{text_body}{summary_prefix}{summary}",
                "input_vars": {
                    "summary_prefix": "\nHere is a summary of the response to this comment:\n"
                },
                "output_template": "{prev_output}{query_prefix}{query}{query_suffix}{gen}{sep}",
                "output_vars": {
                    "query_prefix": "The output for the question \"",
                    "query_suffix": "\" is:\n",
                    "sep": "\n\n"
                }
            },
            {
                "query": "Tell me what this reply is talking about and what the author probably feels about the subject.",
                "depth": 1,
                "input_template": "{text_body}{summary_prefix}{summary}",
                "input_vars": {
                    "summary_prefix": "\nHere is a summary of the response to this comment:\n"
                },
                "output_template": "{prev_output}{query_prefix}{query}{query_suffix}{gen}{sep}",
                "output_vars": {
                    "query_prefix": "The output for the question \"",
                    "query_suffix": "\" is:\n",
                    "sep": "\n\n"
                }
            },
        ], 
        prompt_type: str = "role",
        children_per_summary: int = 5,
        model_params: dict = {}, 
        graph: bool = True
    ):
        self.children_per_summary = children_per_summary 
        self.graph = graph

        self.llm = _ModelWrapper(model_name=model_name, model_origin=model_origin, model_params=model_params)
        
        #Convert summarizer_list into wrapped agents
        self.summarizer_list = self._initialize_qlist(summarizer_list, prompt_type)
        #NOTE: add error checking here.  Make sure that the len of _get_summarizer(1) and _get_summarizer(0) are both >=1
        self.agent_list = self._initialize_qlist(question_list, prompt_type)
        logger.info("inference object initialized")
    
    def _split_into_batches(self, target: list, batch_size: int):
        """Takes a list, and splits into a list of sublists of the origional list each with a size of batch_size"""
        batches = [target[i:i + batch_size] for i in range(0, len(target), batch_size)]
        return batches
    
    def _initialize_qlist(self, input: list[dict], prompt_type: str):
        #Sets the agent objects according to user specifications
        wrapped_list = []
        for query in input:
            logger.debug(f"setting agent object for question: '{query['query']}'")

            if query.get("depth") < -1:
                raise ValueError(f"The question {query} was set to an incorrect value")
            wrapped_list.append(_Agent(query, prompt_type))
        
        #If the number of questions passed in equals the number of questions being returned, return the output
        if len(input) == len(wrapped_list):
            logger.debug(f"Call to function _initialize_qlist wrapped questions: {len(wrapped_list)}/{len(input)}")
            return wrapped_list
        else:
            logger.error("call to _initialize_qlist failed!")
            raise ValueError(f"Call to function _initialize_qlist wrapped questions: {len(wrapped_list)}/{len(input)}")

    def _get_by_depth(self, agent_object_list: list[_Agent], depth: int, name: str = "agent"):
        current_depth = depth
        while True:
            agent_objects = [a for a in agent_object_list if a.depth == current_depth]
            if agent_objects:
                return agent_objects
            current_depth -= 1
            if current_depth < -1:
                raise ValueError(f"No {name}s found for depth {depth}")
    
    def _do_agent_processing(self, current_agents: list, tree_object, top_stack_id: str, prev_summary: str):
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
            text = a.to_input_format({
                "text_body": top_stack_node.data.body,
                "summary": prev_summary
            })
            #Error Checking
            if text == "":
                logger.error(f"\nAgent input formatting failed, empty string!")
            
            gen = self.llm.generate(text, a)
            output = a.to_output_format({
                "prev_output": output,
                "query": a.query,
                "gen": gen
            })
            if output == "":
                logger.warning("Output returned an empty string in agent processing loop!")
        return output

    def _do_summary_processing(self, current_holding: list, summarizer: _Agent):
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
                input = summarizer.to_input_format({"text": batch_text})

                gen = self.llm.generate(input, summarizer)

                output = summarizer.to_output_format({
                    "prev_output": output,
                    "gen": gen
                })
            return output
        else:
            logger.warning("_do_summary_processing was called without children to summarize!  May indicate a post without comments")

    def _do_summary_and_agent(self, tree):
        """
        This function contains the main logic of the script, and handles the loop that traverses through each comment
        tree until the the stack "output_stack" has been depleted down to the root-node.

        Args:
        tree -- The tree containing the conversation data nodes.
        
        Returns:
        final_output -- A list of string reports, with one entry per summarizer with depth of -1
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
                top_stack_node = tree.get_node(output_stack[-1]).data

                logger.debug(f"processing for {output_stack[-1]}.  Children: {len(current_holding)}")
                if self.graph: g.update(top_stack_node.depth)

                children_depth_summarizer = self._get_by_depth(self.summarizer_list, top_stack_node.depth + 1, name="summarizer")[0]
                summary = self._do_summary_processing(current_holding, children_depth_summarizer)

                #if the current top-of-stack is root, return the result
                if output_stack[-1] == tree.root:
                    final_summarizers = self._get_by_depth(self.summarizer_list, -1, name="summarizer")
                    final_output = []
                    for s in final_summarizers:
                        final_input = s.to_input_format({
                            "root": top_stack_node.body,
                            "comment_summaries": summary
                        })

                        final_gen = self.llm.generate(final_input, s)

                        final_output.append(s.to_output_format({
                            "gen": final_gen
                        }))
                    return final_output

                #Clear the now-used entry in temp-holding
                try:
                    del temp_holding[output_stack[-1]]
                except KeyError:
                    logger.debug(f"Node {output_stack[-1]} has no children, skipping tempholding slot clearing")

                #With context from summary, apply depth-appropriate agent(s) to top-of-stack node
                current_agents = self._get_by_depth(self.agent_list, top_stack_node.depth)
                current_agent_output = self._do_agent_processing(current_agents, tree, output_stack[-1], summary)             
                
                #append output from agents to the temp_holding level keyed to the new top-of-stack
                temp_holding[top_stack_node.parent_id].append(current_agent_output)

                if current_agent_output == "":
                    print(f"current_agent_output is an empty string for node {output_stack[-1]} and was added as a child of node {top_stack_node.parent_id}")   

                #Remove completed node from top of the stack
                output_stack.pop()

        #If the while loop completes, then the return statement never triggered
        raise Exception("Return statement never triggered, likely a problem with tree initialization!")

    def process_thread(self, data, data_type: str):
        """
        This high-level function triggers the inference of a passed in conversation, by loading the conversation into
        a treelib structure then passing data to _do_summary_and_agent for logic handling
        
        Args:
        data -- the conversation thread.  No data type hint is defined because unification of datatypes is handled lower in the logic.
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