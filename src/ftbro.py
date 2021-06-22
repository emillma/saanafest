import numpy as np
import sys
from ftbrobackend import FtBroBackend
from fightertwister import to_range, Encoder, ft_colors
import json
from pathlib import Path
import time
import logging


def toggle_2_state(self: Encoder, ts):
    self.set_state(self._state ^ 1)
    self.set_color(ft_colors.red if self._state else self._default_color)


def toggle_signal_shape(self: Encoder, ts):
    self.set_property('sigshape',
                      (self.get_property('sigshape') + 1) % 3)
    print(self.get_property('sigshape'))


def set_filter_low(self: Encoder, ts):
    self.set_property('freq_low', 80 * np.exp(self.value*np.log(3000/80)))
    logging.info(self.get_property('freq_low'))


def set_filter_high(self: Encoder, ts):
    self.set_property('freq_high', 80 * np.exp(self.value*np.log(3000/80)))
    logging.info(self.get_property('freq_high'))


def set_controlled_pitch(self: Encoder, ts):
    self.set_property('controlled_pitch', 80 *
                      np.exp(self.value*np.log(3000/80)))
    logging.info(self.get_property('controlled_pitch'))


class FtBro(FtBroBackend):
    def __init__(self):
        super().__init__()
        self.load()
        for node in self.nodes:

            params = node.get_property('params')

            params[0, 3].register_cb_press(toggle_2_state)

            params[1, 3].register_cb_hold(toggle_2_state)
            params[1, 3].register_cb_click(toggle_signal_shape)
            params[1, 3].register_cb_encoder(set_controlled_pitch)
            params[1, 3].set_property('sigshape', 0)
            set_controlled_pitch(params[1, 3], None)

            params[0, 0].register_cb_encoder(set_filter_low)
            params[0, 0].set_property('freq_low', 0)
            set_filter_low(params[0, 0], None)

            params[1, 0].register_cb_encoder(set_filter_high)
            params[1, 0].set_property('freq_high', 0)
            set_filter_high(params[1, 0], None)

        time.sleep(0.5)

    def get_shift_rate(self, node_idx):
        enc = self.nodes.ravel()[node_idx].get_property('params')[0, 3]
        mean_duration = to_range(enc.value, 0.1, 20)
        return [enc._state, mean_duration, enc.value]

    def get_contolled_tone(self, node_idx):
        param = self.nodes.ravel()[node_idx].get_property('params')[1, 3]
        mode = param._state
        tone = param.get_property('controlled_pitch')
        return [mode, param.get_property('sigshape'), tone]

    def get_node_values(self, i):
        node = self.nodes[np.unravel_index(i, self.nodes.shape)]
        return node.value, node.get_property('params').value

    def save(self, file_name=None):
        state_dict = {}
        for i, selector in enumerate(self.selectors.ravel()):
            selector_dir = {}
            for key in selector._setable_propery_keys:
                # if type(getattr(selector, key)) not in [int, float, bool]:
                selector_dir[key] = getattr(selector, key)

            for j, param in enumerate(selector.get_property('params')):
                param_dir = {}
                for key in param._setable_propery_keys:
                    # if type(getattr(param, key)) not in [int, float, bool]:
                    if key == '_color' and getattr(param, key) == ft_colors.red:
                        a = 1
                    param_dir[key] = getattr(param, key)

                selector_dir[f'param{j:02d}'] = param_dir
            state_dict[f'selector{i:02d}'] = selector_dir

        fname = file_name or Path(__file__).parents[1].joinpath('state.json')
        with open(fname, 'w') as file:
            json.dump(state_dict, file)

    def load(self, file_name=None):
        file = (Path(file_name) if file_name
                else Path(__file__).parents[1].joinpath('state.json'))
        if not file.is_file():
            return
        with open(file, 'r') as file:
            state_dict = json.load(file)
        selectors = self.selectors.ravel()
        for selector, selector_dir in state_dict.items():
            selector = selectors[int(selector[-2:])]
            params = selector.get_property('params').ravel()

            for key, value in selector_dir.items():
                if not key.startswith('param'):
                    getattr(selector, f'set{key}')(value)
                else:
                    param = params[int(key[-2:])]
                    param_dir = value
                    for key, value in param_dir.items():
                        getattr(param, f'set{key}')(value)

    def get_node_filter_low(self, node):
        param = self.nodes.ravel()[node].get_property('params')[0, 0]
        return param.get_property('freq_low')

    def get_node_filter_high(self, node):
        param_low = self.nodes.ravel()[node].get_property('params')[0, 0]
        param_high = self.nodes.ravel()[node].get_property('params')[1, 0]
        return max(param_low.get_property('freq_low')+100,
                   param_high.get_property('freq_high'))


if __name__ == '__main__':
    ft = FtBro()
    with ft:
        ft.save()
        input('press enter to quit')
