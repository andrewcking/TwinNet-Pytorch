import torch
import torch.nn as nn


class TwinNet(nn.Module):
    def __init__(self, num_classes, use_stereo):
        """
        Expirimental TwinNet architecture
        :param num_classes: the number of classes in the datset
        :param use_stereo: use stereo data (if false just use left channel
        """
        super().__init__()  # don't need to pass class in python 3 so NOT super(TwinNet, self).__init()
        self.use_stereo = use_stereo
        if self.use_stereo:
            con_channels = 4096 * 2
        else:
            con_channels = 4096
        self.twin1 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, in_channels=3, out_channels=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=64, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.twin2 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=128, out_channels=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(kernel_size=3, padding=1, in_channels=128, out_channels=128, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.twin3 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, in_channels=128, out_channels=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=256, out_channels=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=256, out_channels=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(kernel_size=3, padding=1, in_channels=256, out_channels=256, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.twin4 = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, in_channels=256, out_channels=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=512, out_channels=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=512, out_channels=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(kernel_size=3, padding=2, in_channels=512, out_channels=512, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=2, in_channels=512, out_channels=512, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size=3, padding=2, in_channels=512, out_channels=512, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(kernel_size=3, padding=2, in_channels=512, out_channels=4096, dilation=2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(kernel_size=1, padding=0, in_channels=4096, out_channels=4096),
            # nn.ReLU(inplace=True),
        )
        self.stereo = nn.Sequential(
            nn.Conv2d(kernel_size=1, padding=0, in_channels=con_channels, out_channels=num_classes),  # double since it was concat

            nn.Conv2d(kernel_size=3, padding=1, in_channels=num_classes, out_channels=64),  # double since it was concat
            nn.PReLU(),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=128),
            nn.PReLU(),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=128, out_channels=256),
            nn.PReLU(),
            nn.Conv2d(kernel_size=3, padding=1, in_channels=256, out_channels=256),  # number of output channels here needs to match output of twin3
            nn.PReLU(),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.collapse = nn.Sequential(
            nn.Conv2d(kernel_size=3, padding=1, in_channels=64, out_channels=64),
            nn.PReLU(),
            nn.Conv2d(kernel_size=1, padding=0, in_channels=64, out_channels=num_classes),  ####### KERNAL SIZE 3???
        )

    def forward(self, left, right=None):
        """
        Forward Pass
        :param left: left image
        :param right: right image
        :return: individual feature maps for each class
        """
        # Pass in Left image
        left_out1 = self.twin1(left.float())  # .float may be needed for CPU
        left_out2 = self.twin2(left_out1)
        left_out3 = self.twin3(left_out2)
        left_out4 = self.twin4(left_out3)  # 4096 x w x h

        if self.use_stereo:
            # Pass in Right image
            right_out = self.twin1(right.float())  # .float may be needed for CPU runs
            right_out = self.twin2(right_out)
            right_out = self.twin3(right_out)
            right_out = self.twin4(right_out)  # 4096 x w x h

            # Concatenate Feature Maps and then pass into stereo module
            twins = torch.cat([left_out4, right_out], dim=1)  # 8192 x w x h
            twins = self.stereo(twins)  # 256 x w x h
        else:
            twins = self.stereo(left_out4)

        # Concatenate Feature Maps and then pass into stereo module
        score = twins + left_out3  # 256 + 256
        score = self.upsample1(score)  # 128

        # Skip Connections transpose convolution upsampling
        score = score + left_out2  # 128 + 128
        score = self.upsample2(score)  # 64

        score = score + left_out1  # 64 + 64
        score = self.upsample3(score)  # 64

        # Collapse to individual feature maps
        score = self.collapse(score)  # num_classes

        return score
