import torch
import torch.nn as nn


def lRelu(x):
    return torch.max(x * 0.2, x)


class AlphaMetaNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.FC = nn.Linear(20, 1)


    def forward(self,EXIF):

        out = self.FC(EXIF)

        return out



# class AlphaMetaNet(nn.Module):
#     def __init__(self, adaptive: bool = False):
#         super().__init__()
#         self.adaptive = adaptive

#         self.Conv1 = nn.Conv2d(
#             in_channels=3, out_channels=5, kernel_size=3, padding=1, stride=2
#         )
#         self.Conv2 = nn.Conv2d(
#             in_channels=5, out_channels=1, kernel_size=3, padding=1, stride=2
#         )

#         self.FC1 = nn.Linear(100, 20)
#         self.FC2 = nn.Linear(20, 1)

#     def _initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d):
#                 module.weight.data.normal_(0.0, 0.02)
#                 if module.bias is not None:
#                     module.bias.data.normal_(0.0, 0.02)
#             if isinstance(module, nn.ConvTranspose2d):
#                 module.weight.data.normal_(0.0, 0.02)
#             if isinstance(module, nn.Linear):
#                 module.weight.data.normal_(0.0, 0.2)

#     def forward(self, x, EXIF, desiredOutExposure):
#         out = self.Conv1(x)
#         out = lRelu(out)
#         out = self.Conv2(out)
#         out = lRelu(out)
#         out = nn.MaxPool2d(2)
#         out = torch.cat([out,EXIF,desiredOutExposure],1)
#         out = self.FC1(out)
#         out = self.FC2(out)

#         return out