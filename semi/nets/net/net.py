from semi.nets.net.CDC import *
# from AttenBlock import *
import torch.nn.functional as F

dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]


class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DilatedConv(in_dim=in_channels, out_dim=out_channels, k=3, dilation=dilation[0][0]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[0][1]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[0][2])
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DilatedConv(in_dim=in_channels, out_dim=out_channels, k=3, dilation=dilation[1][0]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[1][1]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[1][2]),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DilatedConv(in_dim=in_channels, out_dim=out_channels, k=3, dilation=dilation[2][0]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[2][1]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[2][2]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[2][3]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[2][4]),
            DilatedConv(in_dim=out_channels, out_dim=out_channels, k=3, dilation=dilation[2][5])
        )

    def forward(self, x):
        return self.maxpool_conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.up_conv = nn.Sequential(
        #     nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same'),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same'),
        #
        # )

        self.up_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding='same'),
            # nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same'),
            # nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True),
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


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              )
        self.bn = nn.BatchNorm1d(nOut, eps=1e-5)
        self.gelu = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.gelu(output)

        return output


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.stem1 = nn.Sequential(
            Conv(1, 32, kSize=3, stride=1, padding='same'),
            Conv(32, 32, kSize=3, stride=1, padding='same'),
            Conv(32, 32, kSize=3, stride=1, padding='same'),
        )
        self.fc0 = nn.Conv1d(32, 1, kernel_size=1)
        self.fc1 = nn.Conv1d(64, 1, kernel_size=1)
        self.fc2 = nn.Conv1d(128, 1, kernel_size=1)

        # self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down1(32, 64))
        # self.attn = (Block(dim=64, num_patches=1, num_heads=8))
        self.down2 = (Down2(64, 128))
        self.down3 = (Down3(128, 256))
        self.drop_1 = nn.Dropout(p=0.3)
        self.fc3 = nn.Conv1d(256, 1, kernel_size=1)

        self.up1 = (Up(256, 128))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(64, 32))
        self.drop_2 = nn.Dropout(p=0.15)
        self.outc = (OutConv(32, 1))
        self.bn = nn.BatchNorm1d(1)
        # self.softmax = nn.Softmax(dim=-1)
        self.num_class = 4096
        # self.classifier = nn.Linear(4096, self.num_class)
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1), nn.LeakyReLU(0.2, inplace=True), nn.Linear(1, self.num_class)
        )

    def forward(self, x):

        x1 = self.stem1(x)
        stem_feature = self.fc0(x1)
        self.stem_feature = stem_feature.detach()

        x2 = self.down1(x1)
        down1_feature = self.fc1(x2)
        self.down1_feature = down1_feature.detach()

        # x12 = self.attn(x2)
        x3 = self.down2(x2)
        down2_feature = self.fc2(x3)
        self.down2_feature = down2_feature.detach()

        x4 = self.down3(x3)
        x0 = self.drop_1(x4)
        last_feature = self.fc3(x4)
        self.last_feature = last_feature.detach()

        x = self.up1(x0, x3)
        x = self.drop_2(x)
        x = self.up2(x, x2)
        x = self.drop_2(x)
        x = self.up3(x, x1)
        x = self.drop_2(x)
        x = self.outc(x)

        x_fc = x.view(-1, x.size(2))
        # print(xfc.shape)
        # print('xfc:', x_fc.shape)
        output = x_fc
        result_dict = {'logits': output, 'feat': x_fc}
        # print('feat:', x_fc.shape)
        # print('logits:', output.shape)

        # logits = nn.Softmax(logits[:, -1, :])

        # print('logits:', logits)
        # print('softmax_out:', softmax_out)

        return result_dict


def Lite():
    model = UNet()
    return model

# print(UNet(in_channels=1, n_classes=4096))

# from torchsummary import summary
# import torch
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Lite().to(device)
# summary(model, input_size=(1, 4096))

# import torch
from torchvision.models import resnet18

# input_shape = [4096]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet().to(device)
#
# summary(model, input_size=(1, 4096))

# if __name__=='__main__':
#     input = torch.randn(1, 1, 4096)
#     net=UNet(in_channels=1, n_classes=4096)
#     output = net(input)
#     print(output)
#     torch.save(net,'unet.pth')
#     # torch.onnx.export(net, input,  "unet.onnx", export_params=True, opset_version=10,
#     #               do_constant_folding=True,   input_names = ['input'],   output_names = ['output'],)

# model = UNet()
# for name, param in model.named_parameters():
#     print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)

