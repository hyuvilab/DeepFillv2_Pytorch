import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import torchvision

from network_module import *



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
            ('gatedconv', [opt.in_channels, opt.latent_channels, 5, 1, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels * 2, 3, 2, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            # Bottleneck
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, 2, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, 4, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, 8, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, 16, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            # decoder
            ('transposedgatedconv', [opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('transposedgatedconv', [opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels//2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels//2, opt.out_channels, 3, 1, 1, 1, opt.pad_type, 'none', opt.norm]),
            ('tanh', None)
        ]
        self.refine_conv_start = self.build_parameters(self.coarse_config)
        print('refine_conv_start: {}'.format(self.refine_conv_start))


        self.refine_conv_config = [
            ('gatedconv', [opt.in_channels, opt.latent_channels, 5, 1, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels, 3, 2, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels*2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*2, opt.latent_channels*2, 3, 2, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*2, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, 2, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, 4, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, 8, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, 16, opt.pad_type, opt.activation, opt.norm])
        ]
        self.refine_atten_1_start = self.build_parameters(self.refine_conv_config) + self.refine_conv_start

        self.refine_atten_1_config = [
            ('gatedconv', [opt.in_channels, opt.latent_channels, 5, 1, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels, 3, 2, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels*2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*2, opt.latent_channels*4, 3, 2, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, 'relu', opt.norm])
        ]
        self.refine_atten_2_start = self.build_parameters(self.refine_atten_1_config) + self.refine_atten_1_start

        self.refine_atten_2_config = [
            ('gatedconv', [opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm])
        ]
        self.refine_combine_start = self.build_parameters(self.refine_atten_2_config) + self.refine_atten_2_start

        self.refine_combine_config = [
            ('gatedconv', [opt.latent_channels*8, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*4, opt.latent_channels*4, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('transposedgatedconv', [opt.latent_channels * 4, opt.latent_channels*2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels*2, opt.latent_channels*2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('transposedgatedconv', [opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels, opt.latent_channels//2, 3, 1, 1, 1, opt.pad_type, opt.activation, opt.norm]),
            ('gatedconv', [opt.latent_channels//2, opt.out_channels, 3, 1, 1, 1, opt.pad_type, 'none', opt.norm]),
            ('tanh', None)
        ]
        self.context_attention_start = self.build_parameters(self.refine_combine_config) + self.refine_combine_start

        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                     fuse=True)


    def build_parameters(self, config_list):
        count = 0
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
                count += 4

        return count


    def inner_modules(self, x, vars, name):

        if(name == 'coarse'):
            config = self.coarse_config
            idx = 0
        if(name == 'refine_conv'):
            config = self.refine_conv_config
            idx = self.refine_conv_start
        if(name == 'refine_atten_1'):
            config = self.refine_atten_1_config
            idx = self.refine_atten_1_start
        if(name == 'refine_atten_2'):
            config = self.refine_atten_2_config
            idx = self.refine_atten_2_start
        if(name == 'refine_combine'):
            config = self.refine_combine_config
            idx = self.refine_combine_start

        for (name, param_config) in config:
            if(name == 'gatedconv' or name == 'transposedgatedconv'):
                if(name == 'transposedgatedconv'):
                    scale_factor = 2 if(len(param_config) < 10) else param_config[9]
                    x = F.interpolate(x, scale_factor=scale_factor, mode = 'nearest')
                if(param_config[4] > 0):
                    x = F.pad(x, (param_config[4], param_config[4], param_config[4], param_config[4]))

                conv = F.conv2d(x, vars[idx], vars[idx+1], param_config[3], 0, param_config[5])
                mask = F.conv2d(x, vars[idx+2], vars[idx+3], param_config[3], 0, param_config[5])
                gated_mask = F.sigmoid(mask)
                if(param_config[7] == 'lrelu'):
                    conv = F.leaky_relu(conv, 0.2, inplace=True)
                elif(param_config[7] == 'relu'):
                    conv = F.relu(conv, inplace=True)
                x = conv * gated_mask
                idx += 4
            elif(name == 'tanh'):
                x = F.tanh(x)
        return x
                
        
    def forward(self, img, mask, vars=None):
        if(vars is None):
            vars = self.vars

        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)       # in: [B, 4, H, W]
        first_out = self.inner_modules(first_in, vars, 'coarse')                           # out: [B, 3, H, W]
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)
        #refine_conv = self.refine_conv(second_in)     
        refine_conv = self.inner_modules(second_in, vars, 'refine_conv')
        #refine_atten = self.refine_atten_1(second_in)
        refine_atten = self.inner_modules(second_in, vars, 'refine_atten_1')
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten, offset_flow = self.context_attention(refine_atten, refine_atten, mask_s)
        #refine_atten = self.refine_atten_2(refine_atten)
        refine_atten = self.inner_modules(refine_atten, vars, 'refine_atten_2')
        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        #second_out = self.refine_combine(second_out)
        second_out = self.inner_modules(second_out, vars, 'refine_combine')
        second_out = F.interpolate(second_out, (img.shape[2], img.shape[3]))
        offset_flow = F.interpolate(offset_flow, (img.shape[2], img.shape[3]))
        return first_out, second_out, offset_flow

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.vars = nn.ParameterList()

        self.config = [
            ('conv', [opt.in_channels, opt.latent_channels, 7, 1, 3, opt.pad_type, opt.activation, opt.norm]),
            ('conv', [opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('conv', [opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('conv', [opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('conv', [opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, opt.pad_type, opt.activation, opt.norm]),
            ('conv', [opt.latent_channels * 4, 1, 4, 2, 1, opt.pad_type, 'none', 'none'])
        ]
        self.build_parameters(self.config)


    def build_parameters(self, config_list):
        count = 0
        for (name, param_config) in config_list:
            if(name == 'conv'):
                w = nn.Parameter(torch.ones(param_config[1], param_config[0], param_config[2], param_config[2]))
                torch.nn.init.kaiming_normal_(w)

                # Spectral Normalization
                height = w.data.shape[0]
                width = w.view(height, -1).data.shape[1]
                u = Parameter(w.data.new(height).normal_(0, 1))#, requires_grad=False)      # >> This may be problematic!!
                v = Parameter(w.data.new(width).normal_(0, 1))#, requires_grad=False)
                u.data = l2normalize(u.data)
                v.data = l2normalize(v.data)

                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param_config[1])))
                self.vars.append(u)
                self.vars.append(v)
                count += 4
        return count


    def inner_modules(self, x, vars, config):
        idx = 0
        for (name, param_config) in config:
            if(name == 'conv'):
                if(param_config[4] > 0):
                    x = F.pad(x, (param_config[4], param_config[4], param_config[4], param_config[4]))

                # Spectral Normalization here
                w = vars[idx]
                u = vars[idx+2]
                v = vars[idx+3]
                height = w.data.shape[0]
                v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
                u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
                sigma = u.dot(w.view(height, -1).mv(v))
                w = w / sigma.expand_as(w)

                x = F.conv2d(x, w, vars[idx+1], param_config[3], 0)
                if(param_config[7] == 'lrelu'):
                    x = F.leaky_relu(x, 0.2, inplace=True)
                idx += 4
        return x


    def forward(self, img, mask, vars=None):
        if(vars is None):
            vars = self.vars
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.inner_modules(x, vars, self.config)
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
