import numpy as np
from fightertwister import (FighterTwister, Encoder,
                            Button, ft_colors, EncoderCollection)


class FtBroBackend(FighterTwister):
    def __init__(self):
        super().__init__()

        self.selectors = self.encoders[0, :2, :]
        self.nodes = self.selectors[:, :3]
        self.main_volume = self.selectors[0, 3]
        self.mic_volume = self.selectors[1, 3]

        for selector in self.selectors:
            params = EncoderCollection((2, 4), self)
            params.set_default_color(ft_colors.green, True)

            params.set_property('copy_mode', False)
            selector.set_property('params', params)

        self.selectors.set_value(0.3)
        self.selectors.set_color(self.selectors._default_color)

        self.nodes.set_default_color(ft_colors.blue, True)

        self.main_volume.register_cb_hold(self.volume_or_mic_hold)
        self.main_volume.set_default_color(ft_colors.orange, True)

        self.mic_volume.register_cb_hold(self.volume_or_mic_hold)
        self.mic_volume.set_default_color(ft_colors.magenta, True)

        self.button_params = self.sidebuttons[0:2, 0, 0]
        self.button_inspect = self.sidebuttons[0:2, 1, 0]
        self.button_copy = self.sidebuttons[0, 2, 0]

        self.color_node_selected = ft_colors.cyan
        self.color_copy = ft_colors.yellow

        self.selected_node = self.selectors[0, 0]
        self.encoder_slots[0, 2:] = self.selected_node.get_property('params')
        self.selected_node.set_color(self.color_node_selected)

        self.nodes.register_cb_hold(self.node_hold)
        self.selectors.register_cb_click(self.toggle_onoff)

        self.nodes.register_cb_press(self.node_togle_copy_mode)
        self.nodes.register_cb_dbclick(self.toggle_solo)

        self.button_copy.register_cb_dbclick(self.enable_all_copy)
        self.button_copy.register_cb_click(self.disable_all_copy)

        self.button_params.register_cb_press(lambda *_: self.set_bank(0))
        self.button_inspect.register_cb_press(lambda *_: self.set_bank(1))

    def volume_or_mic_hold(self, node: Encoder, _ts):
        self.selectors.set_color(
            self.selectors._default_color)
        node.set_color(self.color_node_selected)
        self.encoder_slots[0, 2:] = node.get_property('params')
        self.selected_node = node

    def node_hold(self, node: Encoder, _ts):
        self.disable_all_copy(None, None)
        self.selectors.set_color(
            self.selectors._default_color)
        self.selected_node = node
        self.selected_node.set_color(self.color_node_selected)
        self.encoder_slots[0, 2:] = node.get_property('params')

    def node_togle_copy_mode(self, node: Encoder, ts):
        if node is not self.selected_node and self.button_copy.pressed:
            if not node.get_property('copy_mode'):
                node.set_property('copy_mode', True)
                node.set_color(self.color_copy)
                node.set_property('params', EncoderCollection(
                                  self.selected_node.get_property('params')))
            else:
                node.set_property('copy_mode', False)
                node.set_color(node._default_color)
                node.set_property('params', EncoderCollection(
                                  node.get_property('params').copy()))

    def enable_all_copy(self, button: Button, ts):
        for node in [n for n in self.nodes if n is not self.selected_node]:
            node.set_property('copy_mode', True)
            node.set_color(self.color_copy)
            node.set_property('params', EncoderCollection(
                              self.selected_node.get_property('params')))

    def disable_all_copy(self, button: Button, ts):
        for node in [n for n in self.nodes if n is not self.selected_node]:
            node.set_property('copy_mode', False)
            node.set_color(node._default_color)
            node.set_property('params', EncoderCollection(
                              node.get_property('params').copy()))

    def toggle_onoff(self, node: Encoder, ts):
        if not self.button_copy.pressed:
            node.set_on(node._on ^ 1)

    def toggle_solo(self, node: Encoder, ts):
        node.set_on(1)
        [i.set_on(0) for i in self.nodes if i is not node]


if __name__ == '__main__':
    ft = FtBroBackend()
    with ft:
        input('press enter to quit')
