import torch
import torch.nn as nn
import torchvision.models as models

RESNETS = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}


class ResnetEncoder(nn.Module):
    """A ResNet that handles multiple input images and outputs skip connections"""

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super().__init__()

        if num_layers not in RESNETS:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = RESNETS[num_layers](pretrained)

        # Up until this point self.encoder handles 3 input channels.
        # For pose estimation we want to use two input images,
        # which means 6 input channels.
        # Extend the encoder in a way that makes it equivalent
        # to the single-image version when fed with an input image
        # repeated num_input_images times.
        # Further Information is found in the appendix Section B of:
        # https://arxiv.org/pdf/1806.01260.pdf
        # Mind that in this step only the weights are changed
        # to handle 6 (or even more) input channels
        # For clarity the attribute "in_channels" should be changed too,
        # although it seems to me that it has no influence on the functionality
        self.encoder.conv1.weight.data = self.encoder.conv1.weight.data.repeat(
            (1, num_input_images, 1, 1)
        ) / num_input_images

        # Change attribute "in_channels" for clarity
        self.encoder.conv1.in_channels = num_input_images * 3  # Number of channels for a picture = 3

        # Remove fully connected layer
        self.encoder.fc = None

        if num_layers > 34:
            self.num_ch_enc = (64, 256,  512, 1024, 2048)
        else:
            self.num_ch_enc = (64, 64, 128, 256, 512)

    def forward(self, l_0):
        l_0 = self.encoder.conv1(l_0)
        l_0 = self.encoder.bn1(l_0)
        l_0 = self.encoder.relu(l_0)

        l_1 = self.encoder.maxpool(l_0)
        l_1 = self.encoder.layer1(l_1)

        l_2 = self.encoder.layer2(l_1)
        l_3 = self.encoder.layer3(l_2)
        l_4 = self.encoder.layer4(l_3)

        return (l_0, l_1, l_2, l_3, l_4)
