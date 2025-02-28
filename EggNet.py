import torch
import torch.nn as nn

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
    

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class MS_Conv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride):
        super(MS_Conv, self).__init__()
        self.in_channels = in_channels
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                Conv(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False, bn_acti=True),
            )
            for kernel_size in kernel_sizes
        ])
    
    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)

        return outputs
    

class LowResolutionBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LowResolutionBranch, self).__init__()
        self.mid_channels = in_channels // 2
        self.dwconv = Conv(in_channels, in_channels, 3, 2, 1, groups=in_channels, bn_acti=True)
        self.pwconv_1 = Conv(in_channels, self.mid_channels, 1, 1, 0, groups=self.mid_channels)
        self.hardswish = nn.Hardswish()
        self.pwconv_2 = Conv(self.mid_channels, out_channels, 1, 1, 0, groups=self.mid_channels)  
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv_1(x)
        x = self.hardswish(x)
        x = self.pwconv_2(x)
        x = channel_shuffle(x, groups=self.mid_channels)
        x = self.upsample(x)
        return x


class HighResolutionBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HighResolutionBranch, self).__init__()
        self.dwconv = Conv(in_channels, out_channels, 3, 1, 1, groups=in_channels, bn_acti=True)
        
    def forward(self, x):
        x = self.dwconv(x)

        return x


class ISA_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ISA_Block, self).__init__()
        self.low_res_branch = LowResolutionBranch(in_channels, out_channels)
        self.high_res_branch = HighResolutionBranch(in_channels, out_channels)
        
    def forward(self, x):
        shortcut = x
        low_res_output = self.low_res_branch(x)
        high_res_output = self.high_res_branch(x)
        output = low_res_output + high_res_output + shortcut

        return output
    

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1, groups=nIn)
        self.conv1x1 = Conv(nConv, nConv, kSize=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)
        output = self.conv1x1(output)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool],1)

        output = self.bn_prelu(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn_prelu = BNPReLU(noutput)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn_prelu(output)

        return output


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    
    return x


class MS_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super(MS_Block, self).__init__()
        self.ms_dwconv = MS_Conv(in_c, [1,3,5], stride=1)
        self.pwconv = Conv(in_c, out_c, 1, 1, 0, bn_acti=True)

    def forward(self, x):
        dwconv_outs = self.ms_dwconv(x)
        dout = 0
        for dwout in dwconv_outs:
            dout = dout + dwout
        out = channel_shuffle(dout, groups=4)
        out = self.pwconv(out)
        out = out + x

        return out  


class LFE_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LFE_Block, self).__init__()
        split_channels = in_channels // 4

        self.conv3x3 = Conv(split_channels, split_channels, 3, 1, 1, bn_acti=True)

        self.conv3x1 = nn.Conv2d(split_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0),groups=split_channels)
        self.conv1x3 = nn.Conv2d(out_channels, split_channels, kernel_size=(1, 3), stride=1, padding=(0, 1),groups=split_channels)
        self.conv1x1_1 = Conv(split_channels, split_channels, 1, 1, 0, bn_acti=True)


        self.conv5x1 = nn.Conv2d(split_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2, 0),groups=split_channels)
        self.conv1x5 = nn.Conv2d(out_channels, split_channels, kernel_size=(1, 5), stride=1, padding=(0, 2),groups=split_channels)
        self.conv1x1_2 = Conv(split_channels, split_channels, 1, 1, 0, bn_acti=True)

        self.conv7x1 = nn.Conv2d(split_channels, out_channels, kernel_size=(7, 1), stride=1, padding=(3, 0),groups=split_channels)
        self.conv1x7 = nn.Conv2d(out_channels, split_channels, kernel_size=(1, 7), stride=1, padding=(0, 3),groups=split_channels)
        self.conv1x1_3 = Conv(split_channels, split_channels, 1, 1, 0, bn_acti=True)

        self.pwconv = Conv(in_channels, out_channels, 1, 1, 0, bn_acti=True)

    def forward(self, x):
        
        shortcut = x  
        split_x = torch.chunk(x, 4, dim=1)

        out0 = self.conv3x3(split_x[0])  

        out1 = self.conv3x1(split_x[1])  
        out1 = self.conv1x3(out1)        
        out1 = self.conv1x1_1(out1)

        out2 = self.conv5x1(split_x[2])  
        out2 = self.conv1x5(out2)       
        out2 = self.conv1x1_2(out2)       

        out3 = self.conv7x1(split_x[3])
        out3 = self.conv1x7(out3)      
        out3 = self.conv1x1_3(out3)       

        output = torch.cat([out0, out1, out2, out3], dim=1)
        output = output + shortcut

        output = self.pwconv(output)
        
        output = channel_shuffle(output, groups=4)
        
        return output
    

class EggNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        self.stem = nn.Sequential(
            Conv(3, 16, 3, 2, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True, groups=16),
            Conv(16, 16, 1, 1, padding=0, bn_acti=True),
            MS_Block(16, 16),
            BNPReLU(16)
        )

        self.encoder_1 = nn.Sequential(
            ISA_Block(16, 16),
            BNPReLU(16),
            DownSamplingBlock(16, 32)
        )

        self.encoder_2 = nn.Sequential(
            ISA_Block(32, 32),
            BNPReLU(32),
            DownSamplingBlock(32, 64)
        )

        self.decoder_2 = nn.Sequential(
            ISA_Block(64, 64),
            BNPReLU(64),
            UpsamplerBlock(64, 32)
        )

        self.decoder_1 = nn.Sequential(
            ISA_Block(32, 32),
            BNPReLU(32),
            UpsamplerBlock(32, 16)
        )

        self.local_conv = Conv(16, 32, 3, 1, 1, bn_acti=True, groups=16)
        self.local_pwconv = Conv(32, 32, 1, 1, 0)
        self.lfe = LFE_Block(32, 16)

        self.act = BNPReLU(16)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(16, classes, 2, stride=2, padding=0, output_padding=0, bias=True)
        )

    def forward(self, x):

        x = self.stem(x)

        e_1 = self.encoder_1(x)
        e_2 = self.encoder_2(e_1)
        d_2 = self.decoder_2(e_2)
        d_1 = self.decoder_1(d_2)

        local_feature = self.local_conv(x)
        local_feature = self.local_pwconv(local_feature)
        local_feature = self.lfe(local_feature)

        output = self.act(d_1 + local_feature)
        out = self.seg_head(output)

        return out
    
