import numpy as np
import sys
from ftbrobackend import FtBroBackend
from fightertwister import to_range, Encoder, ft_colors


def toggle_note_shift_mode(self: Encoder, ts):
    self.set_property('mode',
                      self.get_property('mode') ^ 1)
    if self.get_property('mode'):
        self.set_color(ft_colors.red)
    else:
        self.set_color(ft_colors.green)


def toggle_response_mode(self: Encoder, ts):
    self.set_property('mode',
                      self.get_property('mode') ^ 1)
    if self.get_property('mode'):
        self.set_color(ft_colors.red)
    else:
        self.set_color(ft_colors.green)


def toggle_signal_shape(self: Encoder, ts):
    self.set_property('sigshape',
                      (self.get_property('sigshape') + 1) % 3)
    print(self.get_property('sigshape'))


class FtBro(FtBroBackend):
    def __init__(self):
        super().__init__()
        for node in self.nodes:
            params = node.get_property('params')

            params[0, 3].register_cb_press(toggle_note_shift_mode)
            params[0, 3].set_property('mode', 0)

            params[1, 3].register_cb_hold(toggle_note_shift_mode)
            params[1, 3].register_cb_click(toggle_signal_shape)
            params[1, 3].set_property('mode', 0)
            params[1, 3].set_property('sigshape', 0)

    def __enter__(self):
        super().__enter__()

    def get_shift_rate(self, node_idx):
        enc = self.nodes.ravel()[node_idx].get_property('params')[0, 3]
        return [enc.get_property('mode'), enc.value]

    def get_sine(self, node_idx):
        enc = self.nodes.ravel()[node_idx].get_property('params')[1, 3]
        mode = enc.get_property('mode') * (enc.get_property('sigshape') + 1)
        return [mode, enc.value]

    def get_node_values(self, i):
        node = self.nodes[np.unravel_index(i, self.nodes.shape)]
        return node.value, node.get_property('params').value

    def save(self):
        save_dict = {}
        for i, selector in enumerate(self.selectors.ravel()):
            selector_dir = {}
            for key in selector._setable_propery_keys:
                selector_dir[key] = getattr(selector, key)

            for j, param in enumerate(selector.get_property('params')):
                param_dir = {}
                for key in param._setable_propery_keys:
                    param_dir[key] = getattr(param, key)

                selector_dir[f'param{j:02d}'] = param_dir
            save_dict[f'selector{i:02d}'] = selector_dir
        pass


if __name__ == '__main__':
    ft = FtBro()
    with ft:
        ft.save()
        input('press enter to quit')
