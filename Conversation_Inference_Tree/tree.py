from treelib import Node, Tree
from .reddit_wrapper import _RedditWrapper

from .logger import logger

class _Tree:
    """
    This class is used for turning reddit conversation data into a usable treelib object.

    Methods:
        _recursive_node: A private function to be called by __init__.  Recursively adds an 
                         an entered node's child nodes to the tree.  Utilizes leftmost traversal.
    Args:
        raw_submission: Takes the post and replies to be analyzed.  Currently only takes 
                        json data made by a praw object.  In the future, this will also
                        take praw objects and psaw objects directly.
    """

    def __init__(self, raw_submission):
        self.tree = Tree()
        #NOTE: try changing to dictionary for fast look-up
        self.wrapped_comments = []

        # raw_submission.comments.replace_more(limit=None) NOTE: Figure out if this is still needed

        # Pull comments out of the submission object into a wrappable list
        if isinstance(raw_submission, dict):
            comments = raw_submission.get("comments", [])
        else:
            comments = raw_submission.comments        

        #this will become the root node
        submission = _RedditWrapper(raw_submission)
        #get reddit_wrappers for all comments
        for comment in comments:
            self.wrapped_comments.append(_RedditWrapper(comment))
        logger.debug("thread converted to wrapper objects")

        # Add root node (submission itself)
        self.tree.create_node("root-node", submission.id, data=submission)
        self._recursive_node(submission, submission.id)
        logger.debug("recursion to add wrapper objects to tree complete")

    def _get_children(self, parent_id):
        """Takes a comment's id, and retrieves all the comment objects who have that id as their parent_id attribute as a list"""
        #NOTE: Considering changing to a parent lookup dictionary for wrapped_comments to avoid exponential compute costs
        children_comments = [c for c in self.wrapped_comments if c.parent_id == parent_id]
        return children_comments
    
    #sets the subcomments of entry as child nodes, and repeats the chain
    #does not handle setting the entry node itself, as that would make setting the root complicated
    def _recursive_node(self, entry, parent_id):
        """
        This function takes a comment object, then recursively adds all of that comment's children
        to the treelib object as child nodes.

        Args:
        entry -- the comment object
        parent_id -- the id of the comment object
        """
        logger.debug(f"doing recursion for: {parent_id}")
        children = self._get_children(parent_id)
        for child in children:
            self.tree.create_node(child.id, child.id, parent=entry.id, data=entry)
            
            self._recursive_node(child, child.id)