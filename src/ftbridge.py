from fightertwister import FtBro
import numpy as np


class FtBridge(FtBro):
    def __init__(self):
        super().__init__()

    def node_values(self, i):
        node = self.nodes[np.unravel_index(i, (2, 3))]
        value = node.value
        params = node.get_property('params').value
        return value, params
