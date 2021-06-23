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


def set_node_alpha(self: Encoder, ts):
    self.set_property('node_alpha', 0.001 *
                      np.exp(self.value*np.log(0.5/0.001)))
    logging.info(self.get_property('node_alpha'))


def set_mic_alpha(self: Encoder, ts):
    self.set_property('mic_alpha', 0.001 *
                      np.exp(self.value*np.log(0.5/0.001)))
    logging.info(self.get_property('mic_alpha'))


class FtBro(FtBroBackend):
    def __init__(self):
        super().__init__()
        self.load()
        for i in range(len(self.nodes)):

            filter_low = self.filters_low[i]
            filter_high = self.filters_high[i]
            shift_controller = self.shift_controllers[i]
            pitch_controller = self.pitch_controllers[i]

            shift_controller.register_cb_press(toggle_2_state)
            pitch_controller.register_cb_hold(toggle_2_state)

            pitch_controller.register_cb_click(toggle_signal_shape)
            pitch_controller.register_cb_encoder(set_controlled_pitch)
            pitch_controller.set_property('sigshape', 0)
            set_controlled_pitch(pitch_controller, None)

            filter_low.register_cb_encoder(set_filter_low)
            filter_low.set_property('freq_low', 0)
            set_filter_low(filter_low, None)

            filter_high.register_cb_encoder(set_filter_high)
            filter_high.set_property('freq_high', 0)
            set_filter_high(filter_high, None)

            # params[0, 1].register_cb_encoder(set_node_alpha)
            # params[0, 1].set_property('node_alpha', 0)
            # set_node_alpha(params[0, 1], None)

            # params[1, 1].register_cb_encoder(set_mic_alpha)
            # params[1, 1].set_property('node_alpha', 0)
            # set_mic_alpha(params[1, 1], None)

        time.sleep(0.5)

    @property
    def node_params(self):
        return [node.get_property('params') for node in self.nodes]

    @property
    def filters_low(self):
        return [params[0, 0] for params in self.node_params]

    @property
    def filters_high(self):
        return [params[1, 0] for params in self.node_params]

    @property
    def shift_controllers(self):
        return [params[0, 1] for params in self.node_params]

    @property
    def pitch_controllers(self):
        return [params[1, 1] for params in self.node_params]

    def get_shift_rate(self, node_idx):
        shift_controller = self.shift_controllers[node_idx]
        mean_duration = to_range(shift_controller.value, 0.1, 20)
        return [shift_controller._state, mean_duration, shift_controller.value]

    def get_contolled_tone(self, node_idx):
        pitch_controller = self.pitch_controllers[node_idx]
        mode = pitch_controller._state
        tone = pitch_controller.get_property('controlled_pitch')
        return [mode, pitch_controller.get_property('sigshape'), tone]

    def get_node_filter_low(self, node_idx):
        filter_low = self.filters_low[node_idx]
        return filter_low.get_property('freq_low')

    def get_node_filter_high(self, node_idx):
        filter_low = self.filters_low[node_idx]
        filter_high = self.filters_high[node_idx]
        return max(filter_low.get_property('freq_low') + 100,
                   filter_high.get_property('freq_high'))

    def get_node_values(self, i):
        node = self.nodes[np.unravel_index(i, self.nodes.shape)]
        return node.value, node.get_property('params').value

    def get_alphas(self, n=6):
        node_alpha = np.empty(n, np.float32)
        mic_alpha = np.empty(n, np.float32)
        for i, node in enumerate(self.nodes.ravel()[:n]):
            node_alpha[i] = node.get_property(
                'params')[0, 1].get_property('node_alpha')
            mic_alpha[i] = node.get_property(
                'params')[1, 1].get_property('mic_alpha')
        return node_alpha[None, :], mic_alpha

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


if __name__ == '__main__':
    ft = FtBro()
    with ft:
        ft.save()
        input('press enter to quit')
