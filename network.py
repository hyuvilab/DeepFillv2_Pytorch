import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import torchvision

from network_module import *


'''
[*] Explicit network configuration:

'''

def gen_gatedconv2d_config(in_ch, out_ch, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn=False):
    output_var_shape = []
    output_var_shape.append([in_ch, out_ch, kernel_size, kernel_size])
    output_var_shape.append([in_ch, out_ch, kernel_size, kernel_size])












def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()


        self.vars = nn.ParameterList()

        self.coarse_config = [
            ('gatedconv', [opt.in_channels, opt.latent_channels, 5, 1, 2, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels * 2, 3, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            # Bottleneck
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, 2, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, 4, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, 8, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, 16, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            # decoder
            ('transposedgatedconv', [opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('transposedgatedconv', [opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels//2, 3, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels//2, opt.out_channels, 3, 1, 1, opt.pad_type, 'none', opt.norm]),
            ('tanh', None)
        ]
        self.build_parameters(self.coarse_config)



        self._coarse = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, 5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # decoder
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels//2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels//2, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
            nn.Tanh()
        )
        
        self.refine_conv = nn.Sequential(
            GatedConv2d(opt.in_channels, opt.latent_channels, 5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*2, opt.latent_channels*2, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*2, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
        )
        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(opt.in_channels, opt.latent_channels, 5, 1, 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*2, opt.latent_channels*4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = 'relu', norm = opt.norm)
        )
        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm)
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(opt.latent_channels*8, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels*2, opt.latent_channels*2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.latent_channels//2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels//2, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, activation = 'none', norm = opt.norm),
            nn.Tanh()
        )
        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                     fuse=True)


    def build_parameters(self, config_list):
        for (name, param_config) in config_list:
            if(name == 'gatedconv' or name == 'transposedgatedconv'):
                w = nn.Parameter(torch.ones(param_config[1], param_config[0], param_config[2], param_config[2]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param_config[1])))
                w2 = nn.Parameter(torch.ones(param_config[1], param_config[0], param_config[2], param_config[2]))
                torch.nn.init.kaiming_normal_(w2)
                self.vars.append(w2)
                self.vars.append(nn.Parameter(torch.zeros(param_config[1])))


    def coarse(self, x, vars):
        idx = 0
        for (name, param_config) in self.coarse_config:
            if(name == 'gatedconv' or name == 'transposedgatedconv'):
                if(name == 'transposedgatedconv'):
                    scale_factor = 2 if(len(param_config) < 9) else param_config[8]
                    x = F.interpolate(x, scale_factor=scale_factor, mode = 'nearest')
                x = F.pad(x, param_config[4])
                conv = F.conv2d(x, vars[idx], vars[idx+1], param_config[3], 0, param_config[5])
                mask = F.conv2d(x, vars[idx+2], vars[idx+3], param_config[3], 0, param_config[5])
                gated_mask = F.sigmoid(mask)
                if(param_config[7] != 'none'):
                    conv = F.leaky_relu(conv, 0.2, inplace=True)
                x = conv * gated_mask
                idx += 4
            elif(name == 'tanh'):
                x = F.tanh(x)
                idx += 1

        return x
                
        
    def forward(self, img, mask, vars=None):
        if(vars is None):
            vars = self.vars

        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in, vars)                           # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))

        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)
        refine_conv = self.refine_conv(second_in)     
        refine_atten = self.refine_atten_1(second_in)
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)
        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))
        return first_out, second_out

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block6 = Conv2dLayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type = opt.pad_type, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x
