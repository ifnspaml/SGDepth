import torch
import torch.nn as nn


class MultiRes(nn.Module):
    """ Directly generate target-space outputs from (intermediate) decoder layer outputs
    Args:
        dec_chs: A list of decoder output channel counts
        out_chs: output channels to generate
        pp: A function to call on any output tensor
            for post-processing (like e.g. a non linear activation)
    """

    def __init__(self, dec_chs, out_chs, pp=None):
        super().__init__()

        self.pad = nn.ReflectionPad2d(1)

        self.convs = nn.ModuleList(
            nn.Conv2d(in_chs, out_chs, 3)
            for in_chs in dec_chs[::-1]
        )

        self.pp = pp if (pp is not None) else self._identity_pp

    def _identity_pp(self, x):
        return x

    def forward(self, *x):
        out = tuple(
            self.pp(conv(self.pad(inp)))
            for conv, inp in zip(self.convs[::-1], x)
        )

        return out


class MultiResDepth(MultiRes):
    def __init__(self, dec_chs, out_chs=1):
        super().__init__(dec_chs, out_chs, nn.Sigmoid())

        # Just like in the PoseDecoder, where outputting
        # large translations at the beginning of training
        # is harmful for stability outputting large depths
        # at the beginning of training is a source
        # of instability as well. Increasing the bias on
        # the disparity output decreases the depth output.
        for conv in self.convs:
            conv.bias.data += 5


class MultiResSegmentation(MultiRes):
    def __init__(self, dec_chs, out_chs=20):
        super().__init__(dec_chs, out_chs)

