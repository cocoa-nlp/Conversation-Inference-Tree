import sys
import time

class CLIGraph:
    def __init__(self, min_value, max_value, step_value=1, desc='', width=60):
        self.min_value = min_value
        self.max_value = max_value
        self.step_value = step_value
        self.width = width
        self.desc = desc
        self.current_iter = 1
        self.start_time = time.time()
        self.range_size = (max_value - min_value) // step_value + 1

        # Calculate the sidebar width based on max_value character length
        self.max_value_char_count = len(str(max(abs(self.min_value), abs(self.max_value))))
        self.line_graph_width = width - 4 - self.max_value_char_count  # Adjusted for sidebar

        # Initialize the graph with empty spaces
        self.graph_vals = [[' ' for _ in range(self.line_graph_width)] for _ in range(self.range_size)]

        # Create header and footer
        self.header_prefix = '#' * (2 + self.max_value_char_count)
        self.header = f"{self.header_prefix}{desc}#{'-'*(width - len(desc) - 22 - self.max_value_char_count)}#Itr num: "
        self.footer = f"{self.header_prefix}{'-'*self.line_graph_width}#"

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

    def update(self, value):
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

        # Build printable graph
        full_header = f"{self.header}{str(self.current_iter).zfill(7)}#"
        graph_lines = [f"{self.side_bar[i]}{''.join(self.graph_vals[i])}#" for i in range(self.range_size)]

        printable = "\n".join([full_header] + graph_lines + [self.footer])

        # Clear previous output and print updated graph
        lines_to_move_up = 0
        if self.current_iter > 1:
            lines_to_move_up = self.range_size + 1  # number of printed lines (full_header + graph + footer)
        sys.stdout.write(f"\033[{lines_to_move_up}A")
        sys.stdout.write(f'\r{printable}')
        sys.stdout.flush()

        self.current_iter += 1