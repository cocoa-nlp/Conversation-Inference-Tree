from .logger import logger
class _Agent:
    """
    This object stores the data needed to reference a question that needs to be applied to 
    a given depth's textual input, as well as provides a function for formatting a prompt.

    Methods:
        form_prompt: takes any comment's body text and formats it into a llm-ready prompt,
                     while also applying the query to be answered by the llm.  Should be called
                     within the parenteses the llm's generate() function.
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
    def __init__(self, user_content, prompt_type):
        self.query = user_content.get("query")
        self.depth = user_content.get("depth")
        self.order = user_content.get("order", 1)
        #Prompt_type is to be retrieved directly by outside functions
        self.prompt_type = prompt_type

        self.input_format = Formatter(user_content.get("input_template"), user_content.get("input_vars"))
        self.output_format = Formatter(user_content.get("output_template"), user_content.get("output_vars"))

    def to_input_format(self, system_vars: dict):
        formatted = self.input_format._format(system_vars)
        logger.debug(f"to_input_format was called with a dict of length {len(system_vars)}, and returned object of type {type(formatted)}")
        assert isinstance(formatted, str), f"to_input_format function returned object of type {type(formatted)}, not str"
        return formatted
    
    def to_output_format(self, system_vars: dict):
        formatted = self.output_format._format(system_vars)
        logger.debug(f"to_output_format was called with a dict of length {len(system_vars)}, and returned object of type {type(formatted)}")
        assert isinstance(formatted, str), f"to_output_format function returned object of type {type(formatted)}, not str"
        return formatted

    def form_prompt(self, input_text: str):
        
        #create the prompt by bringing toghether the question the agent will ask(query), 
        #and the textual input the query is focused on.
        if self.prompt_type == "role":
            return [
                {"role": "system", "content": self.query},
                {"role": "user", "content": input_text},
            ]
        elif self.prompt_type == "question":
            return f"{self.query}\n{input_text}"
        else:
            raise ValueError(f"Invalid prompt_type '{self.prompt_type}', must be 'question' or 'role'!")

class Formatter:
    """
    This class is a wrapper for python's template functionality.  The wrapper allows users to utilize
    dictionaries to pass in variables for ease of format customization.

    Methods:
    _format(vars) -- used in internal logic to add data into user-templated strings.

    Args:
    template -- a string containing a series of variable declarations within {}.  Depending on the format being
                initialized, will have different variables that are mandatory to be declared by the user.
    user_vars -- A dictionary that matches to template.  MUST contain all variables that are not defined by internal logic.
                MUST NOT contain any variables that ARE supposed to be defined by internal logic.
    """
    def __init__(self, template: str, user_vars: dict):
        self.template = template
        self.user_vars = user_vars
    
    def _format(self, vars: dict):
        """
        This function takes the user-defined template, then "slots in" the data defined by internal logic.

        Args:
        vars -- This is the same as user_vars in the __init__ function.  A dictionary mapping values to be "slotted into"
                the user-defined template.  The user must not define any key-value pairs that overlap with these values.
                The user must ensure that all values within vars are present within the template string.
        
        Returns:
        formatted_string -- a string containing all variables contained within both user_vars and vars, mapped to their 
                            proper places defined by the template.
        """
        try:
            formatted_string = self.template.format(**self.user_vars, **vars)
            return formatted_string
        except KeyError as e: #TODO: add logging functionality.
            raise ValueError(f"Missing placeholder in template: {e}")