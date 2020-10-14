import torch.nn as nn

class PoseDecoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.nl = nn.ReLU()
        self.squeeze = nn.Conv2d(input_channels, 256, 1)
        self.conv_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_3 = nn.Conv2d(256, 6, 1)

        # The original monodepth2 PoseDecoder
        # included a constant multiplication by
        # 0.01 in the forward pass, possibly to
        # make x_angle and x_translation tiny at
        # the beginning of training for stability.
        # In my opinion this hurts performance
        # with weight_decay enabled.
        # Scaling the initial weights should have
        # a similar effect.
        self.conv_3.weight.data *= 0.01
        self.conv_3.bias.data *= 0.01

    def forward(self, x):
        x = self.squeeze(x)
        x = self.nl(x)

        x = self.conv_1(x)
        x = self.nl(x)

        x = self.conv_2(x)
        x = self.nl(x)

        x = self.conv_3(x)
        x = x.mean((3, 2)).view(-1, 1, 1, 6)

        x_angle = x[..., :3]
        x_translation = x[..., 3:]

        return x_angle, x_translation
