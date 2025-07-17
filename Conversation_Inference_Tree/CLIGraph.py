import sys
import time
 
class CLIGraph:
    """
    This class creates a continuously reprinting graph inspired by tqdm to track the progress of a loop.
    The CLIGraph object is to be instantiated outside the target loop, then its update function is to be
    called within the loop.

    Methods:
    update(value) -- a public-facing function to be called within the target loop.  Passing in an integer will cause
                     its value to be reflected within the graph as the next point in the line.  Calling this function
                     will also increment the iteration tracker in the top-right of the graph.
    _update_timer() -- a function that retrieves the current run-time since initialization of the counter.
                     
    Args:
    min_value -- An int defining the inclusive bottom value of the graph.
    max_value -- An int defining the inclusive top value of the graph.
    step_value -- Default 1, defines the step value between rows of the graph.
    desc -- An optional string value that adds a descriptor title to the top-left of the graph.
    width -- Default 60, defines how many chars wide the graph is.  Wider graphs equal longer graph memory.
    """
    def __init__(self, min_value, max_value, step_value=1, desc='', width=60):
        self.min_value = min_value
        self.max_value = max_value
        self.step_value = step_value
        self.width = width
        self.desc = desc
        self.current_iter = 1
        self.start_time = int(time.time())
        self.range_size = (max_value - min_value) // step_value + 1

        # Calculate the sidebar width based on max_value character length
        self.max_value_char_count = len(str(max(abs(self.min_value), abs(self.max_value))))
        self.line_graph_width = width - 4 - self.max_value_char_count  # Adjusted for sidebar

        # Initialize the graph with empty spaces
        self.graph_vals = [[' ' for _ in range(self.line_graph_width)] for _ in range(self.range_size)]

        # Create header and footer
        self.header_prefix = '#' * (2 + self.max_value_char_count)
        self.header = f"{self.header_prefix}{desc}#{'-'*(width - len(desc) - 22 - self.max_value_char_count)}#Itr num: "
        self.footer = f"{self.header_prefix}"

        # Side bar labels
        self.side_bar = []
        for n in range(max_value, min_value - 1, -step_value):
            if n > 0:
                label = f"+{str(n).zfill(self.max_value_char_count)}#"
            elif n == 0:
                label = f"0{str(0).zfill(self.max_value_char_count)}#"
            else:
                label = f"{str(n).zfill(self.max_value_char_count)}#"
            self.side_bar.append(label)

    def _update_timer(self):
        """When called, return formatted time elapsed since CLIGraph initialization as str"""
        elapsed_time = int(time.time()) - self.start_time
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = elapsed_time % 60
        formatted_time = f"\r{hours:02}:{minutes:02}:{seconds:02}"
        return f"elapsed: {formatted_time}"

    def update(self, value):
        """Updates the graph to the next iteration, and places a █ at the row equaling the variable 'value'"""
        # Shift graph left
        for row in self.graph_vals:
            row.pop(0)

        marker = "█"
        # Determine the position of the new value
        if value > self.max_value:
            row_idx = 0  # top row (overflow "^")
            marker = "^"
        elif value < self.min_value:
            row_idx = self.range_size - 1  # bottom row (underflow "v")
            marker = "v"
        else:
            row_idx = (self.max_value - value) // self.step_value

        # Insert new value
        for idx, row in enumerate(self.graph_vals):
            if idx == row_idx:
                row.append(marker)
            else:
                row.append(" ")
        
        #get elapsed time since graph initialization
        elapsed = self._update_timer

        # Build printable graph
        full_header = f"{self.header}{str(self.current_iter).zfill(7)}#"
        graph_lines = [f"{self.side_bar[i]}{''.join(self.graph_vals[i])}#" for i in range(self.range_size)]
        full_footer = f"{self.footer}{'-'*(self.line_graph_width - len(elapsed))}{elapsed}#"

        printable = "\n".join([full_header] + graph_lines + [full_footer])

        # Clear previous output and print updated graph
        lines_to_move_up = 0
        if self.current_iter > 1:
            lines_to_move_up = self.range_size + 1  # number of printed lines (full_header + graph + footer)
        sys.stdout.write(f"\033[{lines_to_move_up}A")
        sys.stdout.write(f'\r{printable}')
        sys.stdout.flush()

        self.current_iter += 1