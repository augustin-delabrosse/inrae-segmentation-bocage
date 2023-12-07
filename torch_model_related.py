# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/attention-unet.py
# https://www.youtube.com/watch?v=fV6CfJb6NDw&ab_channel=IdiotDeveloper
import torch
import torch.nn as nn

class  conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = conv_block(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder_block(3, 32)  # (3, 64)
        self.e2 = encoder_block(32, 64) # (64, 128)
        self.e3 = encoder_block(64, 128) # (128, 256)
        # self.e4 = encoder_block(128, 256) # Remove

        self.b1 = conv_block(128, 256) # (256, 512)

        self.d1 = decoder_block([256, 128], 128) # ([512, 256], 256)
        self.d2 = decoder_block([128, 64], 64) # ([256, 128], 128)
        self.d3 = decoder_block([64, 32], 32) # ([128, 64], 64)
        # self.d4 = decoder_block([32, 32], 32) # Remove
        
        self.output = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        #s4, p4 = self.e3(p2) # remove

        b1 = self.b1(p3) # (p3)

        d1 = self.d1(b1, s3) # (b1, s3)
        d2 = self.d2(d1, s2) # (d1, s2)
        d3 = self.d3(d2, s1) # (d2, s1)
        #d4 = self.d3(d3, s1) # remove

        output = self.output(d3) # (d3)
        
        output = self.sigmoid(output)
        
        return output


# if __name__ == "__main__":
#     x = torch.randn((8, 3, 256, 256))
#     model = attention_unet()
#     output = model(x)
#     print(output.shape)