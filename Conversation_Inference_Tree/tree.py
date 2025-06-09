from treelib import Node, Tree
from .reddit_wrapper import _RedditWrapper

class _Tree:
    def __init__(self, raw_submission):
        self.tree = Tree()
        self.wrapped_comments = []

        # raw_submission.comments.replace_more(limit=None) NOTE: Figure out if this is still needed

        # Pull comments out of the submission object into a wrappable list
        if isinstance(raw_submission, dict):
            top_comments = raw_submission.get("comments", [])
        else:
            top_comments = raw_submission.comments        

        #this will become the root node
        submission = _RedditWrapper(raw_submission)
        #get reddit_wrappers for all comments
        for comment in top_comments:
            self.wrapped_comments.append(_RedditWrapper(comment))

        # Add root node (submission itself)
        self.tree.create_node("root-node", submission.id, data=submission)
        self._recursive_node(submission, submission.id)
    
    #sets the subcomments of entry as child nodes, and repeats the chain
    #does not handle setting the entry node itself, as that would make setting the root complicated
    def _recursive_node(self, entry, parent_id):
        #entry will be a comment object with 0 or more subcomments

        #NOTE: Considering changing to a parent lookup dictionary for wrapped_comments to avoid exponential compute costs
        for child in [i for i in self.wrapped_comments if i.parent_id == parent_id]:
            self.tree.create_node(child.body[:30], child.id, parent=entry.id, data=entry)
            
            self._recursive_node(child, child.id)