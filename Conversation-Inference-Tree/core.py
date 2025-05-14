

class InferenceTree:
    #set llm
    #arguments:
    # - model type (e.g., OPENAI api, huggingface, other api, etc.)
    # - model name
    # - model key
    # - model parameters (e.g., temperature, max tokens, etc.)

    #set summarizer
    #arguments:
    # - summarizer instruction string
    # - max nodes to summarize
    #this is the agent that will summarize the agent commentary between each time step
    #will pass the text input as a list of strings and the instruction string to the general agent function

    #set node agent
    #arguments:
    # - node name
    # - node query string
    # - node instruction string
    # - a way to define where the node will be applied in the comment tree?

    #_general agent --Handles the basic processing of all agents
    #if statement that constructs the final prompt input based on the model type, using input
    #returns output from the model

    #process reddit thread
    #arguments:
    # - data source type
    # - local data file location
    #NOTE: need to know the input structure of the reddit thread.




    pass