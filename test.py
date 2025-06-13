from treelib import Tree, Node

tree = Tree()

tree.create_node("root-node", "rootid", data="test root node")
tree.create_node("depth1", "depth1id", parent="rootid", data="test depth 1 node")
tree.create_node("depth2", "depth2id", parent="rootid", data="test depth 2 node")

tree.show()