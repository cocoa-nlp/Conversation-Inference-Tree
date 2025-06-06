import openai

class _Agent:
    global query
    def __init__(self, query, depth):
        self.query = query
        self.depth = depth
    
    def get_depth(): return _Agent.depth

    def process(prompt, model): #--Handles the basic processing of all agents
        
        #create the prompt by bringing toghether the question the agent will ask(query), 
        # and the textual input the query is focused on.


        try:
            if isinstance(model, dict):
                response = openai.ChatCompletion.create(
                    model=model["model"],
                    message=prompt,
                    config=model["config"]
                )
                return response['choices'][0]['message']['content']
            else:
                response = model(prompt)
                return response#NOTE: check to see if this is the correct object to return
        except Exception as e:
            print(f"Error generating agent output: {e}")        
