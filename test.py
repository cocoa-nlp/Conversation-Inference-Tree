from cligraph import CLIGraph
import time

#Instantiate the graph object outside the loop
graph = CLIGraph(1, 5, desc='Example Graph')
example_nums = [2,2,2,3,3,4,4,5,6,5,4,5,5,4,3,3,3,3,3,2,2,1,2,3,3]

#Loop through the ints one at a time
for num in example_nums:
    #send the current num to the graph to be visualized
    graph.update(num)
    time.sleep(0.5)