class DeltaBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.conv1 = th.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        self.temb_proj = th.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = self.Normalize(out_channels)
        self.conv2 = th.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)

    def Normalize(self, in_channels):
        return th.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def nonlinearity(self, x):
        # swish
        return x * th.sigmoid(x)


    def forward(self, x, temb):
        h = x

        h = self.conv1(h)
        h = h + self.temb_proj(self.nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.conv2(h)

        return h