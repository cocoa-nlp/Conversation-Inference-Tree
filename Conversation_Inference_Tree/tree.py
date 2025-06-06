from treelib import Node, Tree
from .reddit_wrapper import _RedditWrapper

class _Tree:
    def __init__(self, raw_submission):
        self.tree = Tree()

        # raw_submission.comments.replace_more(limit=None) NOTE: Figure out if this is still needed
        submission = _RedditWrapper(raw_submission)

        # Add root node (submission itself)
        self.tree.create_node("root-node", submission.id, data=submission)

        # Start recursion from top-level comments
        if isinstance(raw_submission, dict):
            top_comments = raw_submission.get("comments", [])
        else:
            top_comments = raw_submission.comments

        for top_level_comment in top_comments: #NOTE: make it so only comments with depth of 0 get through
            self._recursive_node(top_level_comment, submission.id)
    
    #sets the subcomments of entry as child nodes, and repeats the chain
    #does not handle setting the entry node itself, as that would make setting the root complicated
    def _recursive_node(self, entry):
        #entry will be a comment object with 0 or more subcomments

        if len(entry.replies) != 0: #NOTE: RedditWrapper does not have replies attribute
            for child in entry.replies:
                self.tree.create_node(child.body[:30], child.id, parent=entry.id, data=entry)
                
                # self.tree = self._recursive_node(child)
                self._recursive_node(child)
