from ..graph.computation_graph import CompGraph
from ..graph.pipe_graph import PipeGraph
from ..option import PipeOption


class ScheduleSolver:
    def __init__(self, comp_graph: CompGraph, option: PipeOption):
        self.comp_graph = comp_graph
        self.option = option
