class _Agent:
    def __init__(self, query: str, depth: int, order: int = 1):
        self.query = query
        self.order = order
        self.depth = depth

    #Used for matching comments to agents in inference_tree.py function process_thread
    def get_depth(self): return self.depth

    def form_prompt(self, input: str):
        #create the prompt by bringing toghether the question the agent will ask(query), 
        #and the textual input the query is focused on.        
        message = [
            {"role": "system", "content": self.query},
            {"role": "user", "content": input},
        ]
        return message

