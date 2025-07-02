from Conversation_Inference_Tree.inference_tree import InferenceTree
def setup_tree():
    return InferenceTree("model", "origin")


#_get_summarizer
#test to make sure that summarizer is correctly detected by .depth == -1
def test_summarizer_retrieval():
    tree  = setup_tree()
    assert tree._get_summarizer().query == "Summarize the text in 150 words or less."

#  -- _split_into_batches --

#test clean splitting into equal-sized batches
def test_clean_equal_split():
    tree = setup_tree()
    assert tree._split_into_batches([1, 2, 3, 4, 5, 6], 2) == [[1, 2], [3, 4], [5, 6]]

#test splitting into a smaller batch for for remainder elements
def test_clean_unequal_split():
    tree = setup_tree()
    assert tree._split_into_batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

#test what happens when the list is smaller than the batch size
def test_list_smaller_than_batch(): 
    tree = setup_tree()
    assert tree._split_into_batches([1, 2], 4) == [[1, 2]]

#test what happens when there is an empty list
def test_empty_list():
    tree = setup_tree()
    assert tree._split_into_batches([], 5) == []