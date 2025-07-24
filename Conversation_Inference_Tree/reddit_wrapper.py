from typing import Union
from praw.models import Comment, Submission

class _RedditWrapper:
    """
    A wrapper object used to store a single comment or submission within a Reddit thread
    in a unified, easy-to-reference format.  Built with ChatGPT

    Args:
        source -- The raw input of the comment or post. Can be one of:
            - A dict (e.g., JSON from previously saved scraping)
            - A praw.models.Comment object
            - A praw.models.Submission object
    """
    
    def __init__(self, source: Union[dict, Comment, Submission]):
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
        elif isinstance(source, Comment):
            self.id = source.id
            self.body = source.body
            self.parent_id = source.parent_id.split("_")[1] if source.parent_id else None
            self.depth = source.depth if hasattr(source, "depth") else -1

        elif isinstance(source, Submission):
            self.id = source.id
            self.body = source.selftext
            self.parent_id = None
            self.depth = -1

        else:
            raise TypeError(f"{type(source)} is an unsupported data type for RedditWrapper")