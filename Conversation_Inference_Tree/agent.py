class _Agent:
    """
    This object stores the data needed to reference a question that needs to be applied to 
    a given depth's textual input, as well as provides a function for formatting a prompt.

    Methods:
        form_prompt: takes any comment's body text and formats it into a llm-ready prompt,
                     while also applying the query to be answered by the llm.  Should be called
                     within the parentesis the llm's generate() function.
        get_depth: currently unused function to retrieve agent depth.

    Args:
        query: the question that the agent will ask.
        order: possible future functionality that will allow the user to specify same-depth
               ordering of questions, allowing a query to have the context of an earlier 
               query's output.
        depth: defines at what depth a given agent should be applied.  A depth of -1 signifies
               the summarizer to be used on child node outputs.  A 0 signifies top level comments,
               1s are replies to the comments, 2 are replies to replies, etc...
    """

    def __init__(self, query: str, depth: int, order: int = 1, prompt_type: str = "role"):
        self.query = query
        self.order = order
        self.depth = depth
        self.prompt_type = prompt_type

    # Used for matching comments to agents in inference_tree.py function process_thread
    def get_depth(self): return self.depth

    def form_prompt(self, input: str):
        # create the prompt by bringing toghether the question the agent will ask(query),
        # and the textual input the query is focused on.
        if self.prompt_type == "role":
            return [
                {"role": "system", "content": self.query},
                {"role": "user", "content": input},
            ]
        elif self.prompt_type == "question":
            return f"{self.query}\n{input}"
        else:
            raise ValueError(
                f"prompt_type cannot be {self.prompt_type}, must be 'question' or 'role'!")
