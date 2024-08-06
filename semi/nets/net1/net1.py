import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding='same'),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DoubleConv(in_channels, out_channels)
        )



    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        # input is CH
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        #
        # x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])

        x = x1 + x2
        # print('cat:', x.shape)
        return x



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = (DoubleConv(1, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))

        self.fc1 = nn.Conv1d(64, 1, kernel_size=1)
        self.fc2 = nn.Conv1d(128, 1, kernel_size=1)
        self.fc3 = nn.Conv1d(256, 1, kernel_size=1)
        self.fc4 = nn.Conv1d(512, 1, kernel_size=1)
        # self.down5 = (Down(512, 1024))

        # self.up = (Up(1024, 512))
        self.up0 = (Up(512, 256))
        self.up1 = (Up(256, 128))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(64, 32))

        self.outc = (OutConv(32, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        down1_feature = self.fc1(x2)
        self.down1_feature = down1_feature.detach()

        x3 = self.down2(x2)
        down2_feature = self.fc2(x3)
        self.down2_feature = down2_feature.detach()

        x4 = self.down3(x3)
        down3_feature = self.fc3(x4)
        self.down3_feature = down3_feature.detach()

        x5 = self.down4(x4)
        last_feature = self.fc4(x5)
        self.last_feature = last_feature.detach()

        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        x_fc = logits.view(-1, logits.size(2))
        output = x_fc
        result_dict = {'logits': output, 'feat': x_fc}
        return result_dict

def unet():
    model = UNet()
    return model
#
# from torchsummary import summary
# import torch
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet().to(device)
# summary(model, input_size=(1, 4096))