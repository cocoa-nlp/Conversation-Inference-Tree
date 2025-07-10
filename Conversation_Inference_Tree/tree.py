from treelib import Node, Tree
from .reddit_wrapper import _RedditWrapper
from collections import defaultdict

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
        self.wrapped_comments = defaultdict(list)

        # Pull comments out of the submission object into a wrappable list
        if isinstance(raw_submission, dict):
            comments = raw_submission.get("comments", [])
        else:
            comments = raw_submission.comments        

        #this will become the root node
        submission = _RedditWrapper(raw_submission)
        #get reddit_wrappers for all comments
        for comment in comments:
            wrapped_comment = _RedditWrapper(comment)
            self.wrapped_comments[wrapped_comment.parent_id].append(wrapped_comment)
        logger.debug("thread converted to wrapper objects")

        # Add root node (submission itself)
        self.tree.create_node("root-node", submission.id, data=submission)
        self._recursive_node(submission.id)
        logger.debug("recursion to add wrapper objects to tree complete")

    def _get_children(self, parent_id):
        """Takes a comment's id, and retrieves all the comment objects who have that id as their parent_id attribute as a list"""
        children_comments = self.wrapped_comments.get(parent_id, [])
        return children_comments
    
    #sets the subcomments of entry as child nodes, and repeats the chain
    #does not handle setting the entry node itself, as that would make setting the root complicated
    def _recursive_node(self, parent_id):
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
            self.tree.create_node(child.id, child.id, parent=parent_id, data=child)

            if child.depth < 0:
                logger.error(f"Node {child.id} has invalid depth of {child.depth}!")
                raise ValueError(f"Node {child.id} has invalid depth of {child.depth}!")
            
            self._recursive_node(child.id)