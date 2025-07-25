#can take either a json dictionary entry for a comment, or a praw comment object
#Generated by ChatGPT
class _RedditWrapper:
    """
    A wrapper object used to store a single comment within a thread in an easy-to-reference
    object.

    Args:
    source -- the raw input of the entire conversation.  Currently only takes the json data
              created from saving a praw object, but will directly take praw and psaw objects
              in the future.
    """

    def __init__(self, source):
        if isinstance(source, dict):
            parent_id = source.get('parent_id')
            depth = -1

            #triggers if current comment being stored is the post itself
            if parent_id is not None:
                parent_id = parent_id.split("_")[1]
                depth = source.get('depth')

            self.id = source.get('id')
            self.body = source.get('body')
            if self.body == None:
                self.body = source.get('selftext')
            self.parent_id = parent_id
            self.depth = depth
        
        elif hasattr(source, 'body') and hasattr(source, 'parent_id'):
            # Likely a Comment object
            self.id = source.id
            self.body = source.body
            self.parent_id = source.parent_id
            self.depth = getattr(source, 'depth', None)

        elif hasattr(source, 'title') and hasattr(source, 'selftext'):
            # Likely a Submission object
            self.id = source.id
            self.body = f"{source.title}\n\n{source.selftext}".strip()
            self.parent_id = None
            self.depth = 0

        else:
            raise TypeError(f"{source} is an unsupported data type for RedditWrapper")