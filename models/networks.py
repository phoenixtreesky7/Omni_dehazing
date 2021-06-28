### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import math
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from models.spherenet import SphereConv2D, SphereMaxPool2D
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from models import common
import torch.nn.functional as F
from functools import reduce
from options.train_options import TrainOptions
from models.equi_conv import equi_conv2d
from torch.nn.modules.utils import _pair
from models.OmniDepth_network import ConELUBlock
from numpy import random
opt = TrainOptions().parse()

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_strip=2, n_blocks_global=9, n_width=1, 
             n_height=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     

    if netG == 'p2hed_sc_att_hv':        
        netG = D2HED_SC_AttHV(input_nc, output_nc, ngf, n_strip, n_blocks_global, n_width, n_height, norm_layer)
    elif netG == 'p2hed_sc_att':        
        netG = D2HED_SC_Att(input_nc, output_nc, ngf, n_strip, n_blocks_global, n_width, n_height, norm_layer)
    elif netG == 'p2hed_sc_att_c':        
        netG = D2HED_SC_AttC(input_nc, output_nc, ngf, n_strip, n_blocks_global, n_width, n_height, norm_layer)
    elif netG == 'p2hed_sc':        
        netG = D2HED_SC(input_nc, output_nc, ngf, n_strip, n_blocks_global, n_width, n_height, norm_layer)
    elif netG == 'ph_sc_att':        
        netG = DH_SC_Att(input_nc, output_nc, ngf, n_strip, n_blocks_global, n_width, n_height, norm_layer)
    elif netG == 'p2hed_sc_att_strip':        
        netG = D2HED_SC_Att_Strip(input_nc, output_nc, ngf, n_strip, n_blocks_global, n_width, n_height, norm_layer)
    elif netG == 'omnidepth':        
        netG = D2HED_SC_Att_Strip(input_nc, output_nc, ngf, n_strip, n_blocks_global, n_width, n_height, norm_layer)

    else:
        raise('generator not implemented!')
    #print(netG)
    print('gpu_ids',gpu_ids)
    


    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    #print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
##############################################################################
# D2HED_RefineDep_SC_NoAtt
#---------------------------
# Dep2Deh: depth map added to DehazeGenerator's Encoder and Decoder
# SC: SphereConv2D
# NoAtt: no attention model
# RefineDep: add a Depth Refined network
##############################################################################
class OmniDpethNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=3, n_height=3, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(OmniDpethNetwork, self).__init__()
        self.n_width = n_width
        self.omnidepth = OmniDepth()

    def forward(self, input): 
        # First filter bank
        depth = self.omnidepth(input)
        return depth


class StripSensitive_SFF_Model(nn.Module):
    def __init__(self, input_channel, height=128, reduction=8, strip=2, bias=False, norm_layer=nn.InstanceNorm2d,):
        super(StripSensitive_SFF_Model, self).__init__()

        self.strip = strip
        self.height = height

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        d = max(int(height/reduction),4)
        self.conv_squeeze = nn.Sequential(nn.Conv2d(height, d, 1, padding=0, bias=bias), nn.PReLU())
        u = max(int(height/strip),4)
        self.fcs_f0 = nn.Conv2d(d, u, kernel_size=1, stride=1,bias=bias)
        self.fcs_f1 = nn.Conv2d(d, u, kernel_size=1, stride=1,bias=bias)
        self.fcs_f2 = nn.Conv2d(d, u, kernel_size=1, stride=1,bias=bias)
        self.fcs_f3 = nn.Conv2d(d, u, kernel_size=1, stride=1,bias=bias)

        self.softmax = nn.Softmax(dim=2)
        self.conv_up = nn.Conv2d(u, height, 1, stride=1, groups=int(u/2))
        self.conv_smooth = ConELUBlock(input_channel, input_channel, (3, 3), padding=(1, 1))

    def forward(self, input0, input1, input2, input3):

        input0_trans = torch.transpose(input0, 1, 2)
        input1_trans = torch.transpose(input1, 1, 2)
        input2_trans = torch.transpose(input2, 1, 2)
        input3_trans = torch.transpose(input3, 1, 2)

        feature_fuse_1 = input0_trans+input1_trans+input2_trans+input3_trans
        #print('feature_fuse_1',feature_fuse_1.shape)

        pooling = self.global_avg_pool(feature_fuse_1)
        #print('pooling',pooling.shape,'heeh', pooling.squeeze(-1).unsqueeze(0).shape)
        squeeze = self.conv_squeeze(pooling)
        #print('squeeze',squeeze.shape)

        score_f0 = self.fcs_f0(squeeze)
        score_f1 = self.fcs_f1(squeeze)
        score_f2 = self.fcs_f2(squeeze)
        score_f3 = self.fcs_f3(squeeze)
        #print('score_f0',score_f0.shape)

        score_cat = torch.cat((score_f0, score_f1, score_f2, score_f3),2)
        #print('score_cat',score_cat.shape)
        score_att = self.softmax(score_cat)

        #print('score_att',score_att.shape)
        score_chunk = torch.chunk(score_att, 4, 2)
        #print('score_chunk0',score_chunk[0].shape, 'haha', score_chunk[0].permute(0, 3, 2, 1).shape)

        output_f0 = F.interpolate(score_chunk[0].permute(0, 3, 2, 1), size=[1,self.height]) # mode = 'bilinear'
        #print('output_f0',output_f0.shape)
        output_ff0 = output_f0.permute(0, 3, 2, 1) * input0_trans

        output_f1 = F.interpolate(score_chunk[1].permute(0, 3, 2, 1), size=[1,self.height])
        output_ff1 = output_f1.permute(0, 3, 2, 1) * input1_trans

        output_f2 = F.interpolate(score_chunk[2].permute(0, 3, 2, 1), size=[1,self.height])
        output_ff2 = output_f2.permute(0, 3, 2, 1) * input2_trans

        output_f3 = F.interpolate(score_chunk[3].permute(0, 3, 2, 1), size=[1,self.height])
        output_ff3 = output_f3.permute(0, 3, 2, 1) * input3_trans
        #print('output_ff3',output_ff3.shape)

        output = torch.transpose(output_ff0+output_ff1+output_ff2+output_ff3 ,1,2)
        #print('output',output.shape)
        output = self.conv_smooth(output)

        return output, score_att

#################################
##  (Horizontal) SSConv Block  ##
#################################
class HeightWise_SFF_Model(nn.Module):
    def __init__(self, input_channel, height=160, reduction=4,bias=False, norm_layer=nn.InstanceNorm2d,):
        super(HeightWise_SFF_Model, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        d = max(int(height/reduction),4)
        self.conv_squeeze = nn.Sequential(nn.Conv2d(height, d, 1, padding=0, bias=bias), nn.PReLU())
        self.fcs_f0 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f1 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f2 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f3 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)

        self.sigmoid = nn.Softmax(dim=2)

        self.conv_smooth = ConELUBlock(input_channel, input_channel, (5, 3), padding=(2, 1))

    def forward(self, input0, input1, input2, input3):

        input0_trans = torch.transpose(input0, 1, 2)
        input1_trans = torch.transpose(input1, 1, 2)
        input2_trans = torch.transpose(input2, 1, 2)
        input3_trans = torch.transpose(input3, 1, 2)

        feature_fuse_1 = input0_trans+input1_trans+input2_trans+input3_trans
        #print('feature_fuse_1',feature_fuse_1.shape)

        pooling = self.global_avg_pool(feature_fuse_1)
        #print('pooling',pooling.shape)
        squeeze = self.conv_squeeze(pooling)
        #print('squeeze',squeeze.shape)

        score_f0 = self.fcs_f0(squeeze)
        score_f1 = self.fcs_f1(squeeze)
        score_f2 = self.fcs_f2(squeeze)
        score_f3 = self.fcs_f3(squeeze)
        #print('score_f0',score_f0.shape)

        score_cat = torch.cat((score_f0, score_f1, score_f2, score_f3),2)
        #print('score_cat',score_cat.shape)
        score_att = self.sigmoid(score_cat)
        

        #print('score_att',score_att.shape)
        score_chunk = torch.chunk(score_att, 4, 2)

        output_f0 = score_chunk[0] * input0_trans
        output_f1 = score_chunk[1] * input1_trans
        output_f2 = score_chunk[2] * input2_trans
        output_f3 = score_chunk[3] * input3_trans
        #print('output_f0',output_f0.shape)

        output = torch.transpose(output_f0+output_f1+output_f2+output_f3 + feature_fuse_1,1,2)
        #print('output',output.shape)
        output = self.conv_smooth(output)

        return output, score_att


###############################
##  (Vertical) SSConv Block  ##
###############################
class VerticalWise_SFF_Model(nn.Module):
    def __init__(self, input_channel, height=256, reduction=4,bias=False, norm_layer=nn.InstanceNorm2d,):
        super(VerticalWise_SFF_Model, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        d = max(int(height/reduction),4)
        self.conv_squeeze = nn.Sequential(nn.Conv2d(height, d, 1, padding=0, bias=bias), nn.PReLU())
        self.fcs_f0 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f1 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f2 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f3 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)

        self.sigmoid = nn.Softmax(dim=3)

        self.conv_smooth = ConELUBlock(input_channel, input_channel, (5, 3), padding=(2, 1))

    def forward(self, input0, input1, input2, input3):

        input0_trans = torch.transpose(input0, 1, 3)
        input1_trans = torch.transpose(input1, 1, 3)
        input2_trans = torch.transpose(input2, 1, 3)
        input3_trans = torch.transpose(input3, 1, 3)

        feature_fuse_1 = input0_trans+input1_trans+input2_trans+input3_trans
        #print('feature_fuse_1',feature_fuse_1.shape)

        pooling = self.global_avg_pool(feature_fuse_1)
        #print('pooling',pooling.shape)
        squeeze = self.conv_squeeze(pooling)
        #print('squeeze',squeeze.shape)

        score_f0 = self.fcs_f0(squeeze)
        score_f1 = self.fcs_f1(squeeze)
        score_f2 = self.fcs_f2(squeeze)
        score_f3 = self.fcs_f3(squeeze)
        #print('score_f0',score_f0.shape)

        score_cat = torch.cat((score_f0, score_f1, score_f2, score_f3),3)
        #print('score_cat',score_cat.shape)
        score_att = self.sigmoid(score_cat)
        #print('score_att',score_att.shape)
        score_chunk = torch.chunk(score_att, 4, 3)

        output_f0 = score_chunk[0] * input0_trans
        output_f1 = score_chunk[1] * input1_trans
        output_f2 = score_chunk[2] * input2_trans
        output_f3 = score_chunk[3] * input3_trans
        #print('output_f0',output_f0.shape)

        output = torch.transpose(output_f0+output_f1+output_f2+output_f3 + feature_fuse_1,1,3)
        #print('output',output.shape)
        output = self.conv_smooth(output)

        return output,score_att

###################################################
##  Selective Kernel Convolution (SKConv) Block  ##
###################################################
class ChannleWise_SFF_Model(nn.Module):
    def __init__(self, input_channel, height=256, reduction=4,bias=False, norm_layer=nn.InstanceNorm2d,):
        super(ChannleWise_SFF_Model, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        d = max(int(height/reduction),4)
        self.conv_squeeze = nn.Sequential(nn.Conv2d(height, d, 1, padding=0, bias=bias), nn.PReLU())
        self.fcs_f0 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f1 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f2 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)
        self.fcs_f3 = nn.Conv2d(d, height, kernel_size=1, stride=1,bias=bias)

        self.sigmoid = nn.Softmax(dim=2)

        self.conv_smooth = ConELUBlock(input_channel, input_channel, (5, 3), padding=(2, 1))

    def forward(self, input0, input1, input2, input3):

        #input0_trans = torch.transpose(input0, 1, 3)
        #input1_trans = torch.transpose(input1, 1, 3)
        #input2_trans = torch.transpose(input2, 1, 3)
        #input3_trans = torch.transpose(input3, 1, 3)

        feature_fuse_1 = input0+input1+input2+input3
        #print('feature_fuse_1',feature_fuse_1.shape)

        pooling = self.global_avg_pool(feature_fuse_1)
        #print('pooling',pooling.shape)
        squeeze = self.conv_squeeze(pooling)
        #print('squeeze',squeeze.shape)

        score_f0 = self.fcs_f0(squeeze)
        score_f1 = self.fcs_f1(squeeze)
        score_f2 = self.fcs_f2(squeeze)
        score_f3 = self.fcs_f3(squeeze)
        #print('score_f0',score_f0.shape)

        score_cat = torch.cat((score_f0, score_f1, score_f2, score_f3),2)
        #print('score_cat',score_cat.shape)
        score_att = self.sigmoid(score_cat)
        #print('score_att',score_att.shape)
        score_chunk = torch.chunk(score_att, 4, 2)

        output_f0 = score_chunk[0] * input0
        output_f1 = score_chunk[1] * input1
        output_f2 = score_chunk[2] * input2
        output_f3 = score_chunk[3] * input3
        #print('output_f0',output_f0.shape)

        output = output_f0+output_f1+output_f2+output_f3 + feature_fuse_1
        #print('output',output.shape)
        output = self.conv_smooth(output)

        return output,score_att

###########################
##  OmniDehaze + SKConv  ##
###########################
class D2HED_SC_AttC(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=256, n_height=128, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(D2HED_SC_AttC, self).__init__()
        self.n_width = n_width
        self.n_height = n_height
        activation = nn.ReLU(True) 
        #self.g_depth = OmniDepth()
        self.g_depth_coase = CoaseDepthUNet(input_nc, 1, ngf, n_strip,n_blocks_global,n_width,n_height,norm_layer)

        #self.rwsff_0 = HeightWise_SFF_Model(ngf,height=self.n_height,reduction=4,bias=False,norm_layer=norm_layer)
        #self.rwsff_1 = HeightWise_SFF_Model(ngf*2,height=self.n_height,reduction=4,bias=False,norm_layer=norm_layer)
        #self.vwsff_0 = VerticalWise_SFF_Model(ngf,height=self.n_width,reduction=4,bias=False,norm_layer=norm_layer)
        #self.vwsff_1 = VerticalWise_SFF_Model(ngf*2,height=self.n_width,reduction=4,bias=False,norm_layer=norm_layer)
        self.cwsff_0 = ChannleWise_SFF_Model(ngf, height=ngf,reduction=4,bias=False,norm_layer=norm_layer)
        self.cwsff_1 = ChannleWise_SFF_Model(ngf*2, height=ngf*2,reduction=4,bias=False,norm_layer=norm_layer)

        
        ###### DehazeGenerator #####
        ### feature extractor

        self.extractor_0_0 = ConELUBlock(input_nc, ngf, (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, ngf, (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, ngf, (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, ngf, 7, padding=3)

        self.extractor_1_0 = ConELUBlock(ngf, ngf * 2, (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(ngf, ngf * 2, (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(ngf, ngf * 2, (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(ngf, ngf * 2, 5, padding=2)

        mult = 2
        g_dehaze_e0 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e0 = nn.Sequential(*g_dehaze_e0)

        mult = 4
        g_dehaze_e1 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e1 = nn.Sequential(*g_dehaze_e1)

        mult = 8
        g_dehaze_e2 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e2 = nn.Sequential(*g_dehaze_e2)

        ### resnet blocks
        g_dehaze_t = []
        mult = 2**(3 +1)
        for i in range(n_blocks_global):
            g_dehaze_t += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.g_dehaze_t = nn.Sequential(*g_dehaze_t)
        
        ### decoder         
        mult = 16        
        g_dehaze_d2 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d2 = nn.Sequential(*g_dehaze_d2)

        mult = 8        
        g_dehaze_d1 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d1 = nn.Sequential(*g_dehaze_d1)

        mult = 4        
        g_dehaze_d0 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d0 = nn.Sequential(*g_dehaze_d0)


        final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
        self.final = nn.Sequential(*final)           
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        self.pyramid_enhancer = Dehaze1()

    def forward(self, input): 
        # First filter bank
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        #feature_0_cat = torch.cat((feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)

        #feature_hfuse_0 = self.rwsff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)
        #feature_vfuse_0 = self.vwsff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)
        feature_cfuse_0, sc0 = self.cwsff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)

        ##  record the attention value  ##
        #score_att_np = sc0.cpu().detach().numpy()
        #score_att_np_squeeze = np.squeeze(score_att_np)
        #x = random.randint(100000)
        #np.savetxt('score_att_c/score_att_sc0_'+ str(x) +'.txt', score_att_np_squeeze)

        feature_fuse_0 = feature_cfuse_0# + feature_vfuse_0 + feature_cfuse_0

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_fuse_0)
        feature_1_1 = self.extractor_1_1(feature_fuse_0)
        feature_1_2 = self.extractor_1_2(feature_fuse_0)
        feature_1_3 = self.extractor_1_3(feature_fuse_0)
        #feature_1_cat = torch.cat((feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        #feature_hfuse_1 = self.rwsff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)
        #feature_vfuse_1 = self.vwsff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)
        feature_cfuse_1, sc1 = self.cwsff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)

        ##  record the attention value  ##
        #score_att_np = sc1.cpu().detach().numpy()
        #score_att_np_squeeze = np.squeeze(score_att_np)
        ##x = random.randint(100000)
        #np.savetxt('score_att_c/score_att_sc1_'+ str(x) +'.txt', score_att_np_squeeze)

        feature_fuse_1 = feature_cfuse_1# + feature_vfuse_1 + feature_cfuse_1

        depths = self.g_depth_coase(feature_fuse_1)
        depth = depths[0]

        # encoder
        tmp = torch.cat((feature_fuse_1, depth), 1)
        dehaze_d1 = self.g_dehaze_e0(tmp)  # ds

        depth_ds1 = self.downsample(depth)
        tmp = torch.cat((dehaze_d1,depth_ds1), 1)
        dehaze_d2 = self.g_dehaze_e1(tmp)

        depth_ds2 = self.downsample(depth_ds1)
        tmp = torch.cat((dehaze_d2,depth_ds2), 1)
        dehaze_d3 = self.g_dehaze_e2(tmp)

        dehaze_t = self.g_dehaze_t(dehaze_d3)

        # decoder
        depth_ds3 = self.downsample(depth_ds2)
        tmp = torch.cat((dehaze_t, depth_ds3, dehaze_d3), 1)
        dehaze_us3 = self.g_dehaze_d2(tmp)

        tmp = torch.cat((dehaze_us3,depth_ds2,dehaze_d2), 1)
        dehaze_us2 = self.g_dehaze_d1(tmp)

        tmp = torch.cat((dehaze_us2,depth_ds1,dehaze_d1), 1)
        dehaze_us1 = self.g_dehaze_d0(tmp)

        dehaze_f = self.final(dehaze_us1)

        # pyramid enhancer
        tmp = torch.cat((dehaze_f, input), 1)
        dehaze = self.pyramid_enhancer(tmp)

        return depths, dehaze

###########################
##  OmniDehaze + SSConv  ##
###########################
class D2HED_SC_Att(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=512, n_height=256, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(D2HED_SC_Att, self).__init__()
        self.n_width = n_width
        self.n_height = n_height
        activation = nn.ReLU(True) 
        #self.g_depth = OmniDepth()
        self.g_depth_coase = CoaseDepthUNet(input_nc, 1, ngf, n_strip,n_blocks_global,n_width,n_height,norm_layer)

        self.rwsff_0 = HeightWise_SFF_Model(ngf,height=n_height,reduction=4,bias=False,norm_layer=norm_layer)
        self.rwsff_1 = HeightWise_SFF_Model(ngf*2,height=n_height,reduction=4,bias=False,norm_layer=norm_layer)

        
        ###### DehazeGenerator #####
        ### feature extractor

        self.extractor_0_0 = ConELUBlock(input_nc, ngf, (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, ngf, (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, ngf, (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, ngf, 7, padding=3)

        self.extractor_1_0 = ConELUBlock(ngf, ngf * 2, (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(ngf, ngf * 2, (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(ngf, ngf * 2, (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(ngf, ngf * 2, 5, padding=2)

        mult = 2
        g_dehaze_e0 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e0 = nn.Sequential(*g_dehaze_e0)

        mult = 4
        g_dehaze_e1 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e1 = nn.Sequential(*g_dehaze_e1)

        mult = 8
        g_dehaze_e2 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e2 = nn.Sequential(*g_dehaze_e2)

        ### resnet blocks
        g_dehaze_t = []
        mult = 2**(3 +1)
        for i in range(n_blocks_global):
            g_dehaze_t += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.g_dehaze_t = nn.Sequential(*g_dehaze_t)
        
        ### decoder         
        mult = 16        
        g_dehaze_d2 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d2 = nn.Sequential(*g_dehaze_d2)

        mult = 8        
        g_dehaze_d1 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d1 = nn.Sequential(*g_dehaze_d1)

        mult = 4        
        g_dehaze_d0 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d0 = nn.Sequential(*g_dehaze_d0)


        final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
        self.final = nn.Sequential(*final)           
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        self.pyramid_enhancer = Dehaze1()

    def forward(self, input): 
        # First filter bank
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        #feature_0_cat = torch.cat((feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)
        #print('input',input.shape)

        feature_fuse_0, sh0 = self.rwsff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)

        ## record the attention value  ##
        #score_att_np = sh0.cpu().detach().numpy()
        #score_att_np_squeeze = np.squeeze(score_att_np)
        #x = random.randint(100000)
        #np.savetxt('score_att_h/score_att_sh0_'+ str(x) +'.txt', score_att_np_squeeze)

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_fuse_0)
        feature_1_1 = self.extractor_1_1(feature_fuse_0)
        feature_1_2 = self.extractor_1_2(feature_fuse_0)
        feature_1_3 = self.extractor_1_3(feature_fuse_0)
        #feature_1_cat = torch.cat((feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        feature_fuse_1,sh1 = self.rwsff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)

        ## record the attention value  ##
        #score_att_np = sh1.cpu().detach().numpy()
        #score_att_np_squeeze = np.squeeze(score_att_np)
        ##x = random.randint(100000)
        #np.savetxt('score_att_h/score_att_sh1_'+ str(x) +'.txt', score_att_np_squeeze)

        depths = self.g_depth_coase(feature_fuse_1)
        depth = depths[0]

        # encoder
        tmp = torch.cat((feature_fuse_1, depth), 1)
        dehaze_d1 = self.g_dehaze_e0(tmp)  # ds
        #print('dehaze_d1',dehaze_d1.shape)

        depth_ds1 = self.downsample(depth)
        tmp = torch.cat((dehaze_d1,depth_ds1), 1)
        dehaze_d2 = self.g_dehaze_e1(tmp)
        #print('dehaze_d2',dehaze_d2.shape)

        depth_ds2 = self.downsample(depth_ds1)
        tmp = torch.cat((dehaze_d2,depth_ds2), 1)
        dehaze_d3 = self.g_dehaze_e2(tmp)
        #print('dehaze_d3',dehaze_d3.shape)

        dehaze_t = self.g_dehaze_t(dehaze_d3)
        #print('t',dehaze_t.shape)

        # decoder
        depth_ds3 = self.downsample(depth_ds2)
        #print('depth_ds3',depth_ds3.shape)
        tmp = torch.cat((dehaze_t, depth_ds3, dehaze_d3), 1)
        dehaze_us3 = self.g_dehaze_d2(tmp)

        tmp = torch.cat((dehaze_us3,depth_ds2,dehaze_d2), 1)
        dehaze_us2 = self.g_dehaze_d1(tmp)

        tmp = torch.cat((dehaze_us2,depth_ds1,dehaze_d1), 1)
        dehaze_us1 = self.g_dehaze_d0(tmp)

        dehaze_f = self.final(dehaze_us1)

        # pyramid enhancer
        tmp = torch.cat((dehaze_f, input), 1)
        dehaze = self.pyramid_enhancer(tmp)

        return depths, dehaze

#############################################
##  OmniDehaze + SSConv (stripe size > 1)  ##
#############################################
class D2HED_SC_Att_Strip(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=256, n_height=128, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(D2HED_SC_Att_Strip, self).__init__()
        self.n_width = n_width
        self.n_height = n_height
        self.strip = n_strip
        activation = nn.ReLU(True) 
        #self.g_depth = OmniDepth()
        self.g_depth_coase = CoaseDepthUNet(input_nc, 1, ngf, n_strip, n_blocks_global,n_width,n_height,norm_layer)

        self.sssff_0 = StripSensitive_SFF_Model(ngf, height=n_height, reduction=8, strip=self.strip, bias=False, norm_layer=norm_layer)
        self.sssff_1 = StripSensitive_SFF_Model(ngf*2, height=n_height, reduction=8, strip=self.strip, bias=False, norm_layer=norm_layer)

        
        ###### DehazeGenerator #####
        ### feature extractor

        self.extractor_0_0 = ConELUBlock(input_nc, ngf, (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, ngf, (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, ngf, (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, ngf, 7, padding=3)

        self.extractor_1_0 = ConELUBlock(ngf, ngf * 2, (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(ngf, ngf * 2, (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(ngf, ngf * 2, (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(ngf, ngf * 2, 5, padding=2)

        mult = 2
        g_dehaze_e0 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e0 = nn.Sequential(*g_dehaze_e0)

        mult = 4
        g_dehaze_e1 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e1 = nn.Sequential(*g_dehaze_e1)

        mult = 8
        g_dehaze_e2 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e2 = nn.Sequential(*g_dehaze_e2)

        ### resnet blocks
        g_dehaze_t = []
        mult = 2**(3 +1)
        for i in range(n_blocks_global):
            g_dehaze_t += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.g_dehaze_t = nn.Sequential(*g_dehaze_t)
        
        ### decoder         
        mult = 16        
        g_dehaze_d2 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d2 = nn.Sequential(*g_dehaze_d2)

        mult = 8        
        g_dehaze_d1 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d1 = nn.Sequential(*g_dehaze_d1)

        mult = 4        
        g_dehaze_d0 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d0 = nn.Sequential(*g_dehaze_d0)


        final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
        self.final = nn.Sequential(*final)           
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        self.pyramid_enhancer = Dehaze1()

    def forward(self, input): 
        # First filter bank
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        #feature_0_cat = torch.cat((feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)
        #print('input',input.shape)

        feature_fuse_0, sh0 = self.sssff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)

        #score_att_np = sh0.cpu().detach().numpy()
        #score_att_np_squeeze = np.squeeze(score_att_np)
        #x = random.randint(100000)
        #np.savetxt('score_att_strip4/score_att_sh0_'+ str(x) +'.txt', score_att_np_squeeze)

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_fuse_0)
        feature_1_1 = self.extractor_1_1(feature_fuse_0)
        feature_1_2 = self.extractor_1_2(feature_fuse_0)
        feature_1_3 = self.extractor_1_3(feature_fuse_0)
        #feature_1_cat = torch.cat((feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        feature_fuse_1,sh1 = self.sssff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)

        #score_att_np = sh1.cpu().detach().numpy()
        #score_att_np_squeeze = np.squeeze(score_att_np)
        #np.savetxt('score_att_strip4/score_att_sh1_'+ str(x) +'.txt', score_att_np_squeeze)

        depths = self.g_depth_coase(feature_fuse_1)
        depth = depths[0]

        # encoder
        tmp = torch.cat((feature_fuse_1, depth), 1)
        dehaze_d1 = self.g_dehaze_e0(tmp)  # ds
        #print('dehaze_d1',dehaze_d1.shape)

        depth_ds1 = self.downsample(depth)
        tmp = torch.cat((dehaze_d1,depth_ds1), 1)
        dehaze_d2 = self.g_dehaze_e1(tmp)
        #print('dehaze_d2',dehaze_d2.shape)

        depth_ds2 = self.downsample(depth_ds1)
        tmp = torch.cat((dehaze_d2,depth_ds2), 1)
        dehaze_d3 = self.g_dehaze_e2(tmp)
        #print('dehaze_d3',dehaze_d3.shape)

        dehaze_t = self.g_dehaze_t(dehaze_d3)
        #print('t',dehaze_t.shape)

        # decoder
        depth_ds3 = self.downsample(depth_ds2)
        #print('depth_ds3',depth_ds3.shape)
        tmp = torch.cat((dehaze_t, depth_ds3, dehaze_d3), 1)
        dehaze_us3 = self.g_dehaze_d2(tmp)

        tmp = torch.cat((dehaze_us3,depth_ds2,dehaze_d2), 1)
        dehaze_us2 = self.g_dehaze_d1(tmp)

        tmp = torch.cat((dehaze_us2,depth_ds1,dehaze_d1), 1)
        dehaze_us1 = self.g_dehaze_d0(tmp)

        dehaze_f = self.final(dehaze_us1)

        # pyramid enhancer
        tmp = torch.cat((dehaze_f, input), 1)
        dehaze = self.pyramid_enhancer(tmp)

        return depths, dehaze

##############################
##  OmniDehaze + OmniDepth  ##
##############################
class D2HED_SC(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=3, n_height=3, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(D2HED_SC, self).__init__()
        self.n_width = n_width
        activation = nn.ReLU(True) 
        #self.g_depth = OmniDepth()
        self.g_depth_coase = CoaseDepthUNet(input_nc, 1, ngf, n_strip,n_blocks_global,n_width,n_height,norm_layer)

        
        ###### DehazeGenerator #####
        ### feature extractor

        self.extractor_0_0 = ConELUBlock(input_nc, int(ngf/4), (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, int(ngf/4), (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, int(ngf/4), (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, int(ngf/4), 7, padding=3)

        self.extractor_1_0 = ConELUBlock(ngf, int(ngf/2), (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(ngf, int(ngf/2), (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(ngf, int(ngf/2), (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(ngf, int(ngf/2), 5, padding=2)

        mult = 2
        g_dehaze_e0 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e0 = nn.Sequential(*g_dehaze_e0)

        mult = 4
        g_dehaze_e1 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e1 = nn.Sequential(*g_dehaze_e1)

        mult = 8
        g_dehaze_e2 = [nn.Conv2d(ngf * mult+1, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e2 = nn.Sequential(*g_dehaze_e2)

        ### resnet blocks
        g_dehaze_t = []
        mult = 2**(3 +1)
        for i in range(n_blocks_global):
            g_dehaze_t += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.g_dehaze_t = nn.Sequential(*g_dehaze_t)
        
        ### decoder         
        mult = 16        
        g_dehaze_d2 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d2 = nn.Sequential(*g_dehaze_d2)

        mult = 8        
        g_dehaze_d1 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d1 = nn.Sequential(*g_dehaze_d1)

        mult = 4        
        g_dehaze_d0 = [nn.ConvTranspose2d(ngf * mult * 2 + 1, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d0 = nn.Sequential(*g_dehaze_d0)


        final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
        self.final = nn.Sequential(*final)           
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        self.pyramid_enhancer = Dehaze1()

    def forward(self, input): 
        # First filter bank
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        feature_0_cat = torch.cat(
            (feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_0_cat)
        feature_1_1 = self.extractor_1_1(feature_0_cat)
        feature_1_2 = self.extractor_1_2(feature_0_cat)
        feature_1_3 = self.extractor_1_3(feature_0_cat)
        feature_1_cat = torch.cat(
            (feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        depths = self.g_depth_coase(feature_1_cat)
        depth = depths[0]

        # encoder
        tmp = torch.cat((feature_1_cat, depth), 1)
        dehaze_d1 = self.g_dehaze_e0(tmp)  # ds

        depth_ds1 = self.downsample(depth)
        tmp = torch.cat((dehaze_d1,depth_ds1), 1)
        dehaze_d2 = self.g_dehaze_e1(tmp)

        depth_ds2 = self.downsample(depth_ds1)
        tmp = torch.cat((dehaze_d2,depth_ds2), 1)
        dehaze_d3 = self.g_dehaze_e2(tmp)

        dehaze_t = self.g_dehaze_t(dehaze_d3)

        # decoder
        depth_ds3 = self.downsample(depth_ds2)
        tmp = torch.cat((dehaze_t, depth_ds3, dehaze_d3), 1)
        dehaze_us3 = self.g_dehaze_d2(tmp)

        tmp = torch.cat((dehaze_us3,depth_ds2,dehaze_d2), 1)
        dehaze_us2 = self.g_dehaze_d1(tmp)

        tmp = torch.cat((dehaze_us2,depth_ds1,dehaze_d1), 1)
        dehaze_us1 = self.g_dehaze_d0(tmp)

        dehaze_f = self.final(dehaze_us1)

        # pyramid enhancer
        tmp = torch.cat((dehaze_f, input), 1)
        dehaze = self.pyramid_enhancer(tmp)

        return depths, dehaze


###########################################################################
##  OmniDehaze : Separate Dehazing Module and Depth Eestimation Module   ##
##  (predict both two modules, but cut off the connection between them)  ##
###########################################################################
class DH_SC_Att(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=3, n_height=3, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(DH_SC_Att, self).__init__()
        self.n_width = n_width
        activation = nn.ReLU(True) 
        #self.g_depth = OmniDepth()
        self.g_depth_coase = CoaseDepthUNet(input_nc, 1, ngf, n_strip,n_blocks_global,n_width,n_height,norm_layer)

        self.rwsff_0 = HeightWise_SFF_Model(ngf,height=n_height,reduction=4,bias=False,norm_layer=norm_layer)
        self.rwsff_1 = HeightWise_SFF_Model(ngf*2,height=n_height,reduction=4,bias=False,norm_layer=norm_layer)

        
        ###### DehazeGenerator #####
        ### feature extractor

        self.extractor_0_0 = ConELUBlock(input_nc, ngf, (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, ngf, (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, ngf, (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, ngf, 7, padding=3)

        self.extractor_1_0 = ConELUBlock(ngf, ngf * 2, (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(ngf, ngf * 2, (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(ngf, ngf * 2, (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(ngf, ngf * 2, 5, padding=2)

        mult = 2
        g_dehaze_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e0 = nn.Sequential(*g_dehaze_e0)

        mult = 4
        g_dehaze_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e1 = nn.Sequential(*g_dehaze_e1)

        mult = 8
        g_dehaze_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e2 = nn.Sequential(*g_dehaze_e2)

        ### resnet blocks
        g_dehaze_t = []
        mult = 2**(3 +1)
        for i in range(n_blocks_global):
            g_dehaze_t += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.g_dehaze_t = nn.Sequential(*g_dehaze_t)
        
        ### decoder         
        mult = 16        
        g_dehaze_d2 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d2 = nn.Sequential(*g_dehaze_d2)

        mult = 8        
        g_dehaze_d1 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d1 = nn.Sequential(*g_dehaze_d1)

        mult = 4        
        g_dehaze_d0 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d0 = nn.Sequential(*g_dehaze_d0)


        final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
        self.final = nn.Sequential(*final)           
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        self.pyramid_enhancer = Dehaze1()

    def forward(self, input): 
        # First filter bank
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        #feature_0_cat = torch.cat((feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)

        feature_fuse_0,s = self.rwsff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_fuse_0)
        feature_1_1 = self.extractor_1_1(feature_fuse_0)
        feature_1_2 = self.extractor_1_2(feature_fuse_0)
        feature_1_3 = self.extractor_1_3(feature_fuse_0)
        #feature_1_cat = torch.cat((feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        feature_fuse_1,s = self.rwsff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)

        depths = self.g_depth_coase(feature_fuse_1)
        depth = depths[0]

        # encoder
        #tmp = torch.cat((feature_fuse_1, depth), 1)
        dehaze_d1 = self.g_dehaze_e0(feature_fuse_1)  # ds

        #depth_ds1 = self.downsample(depth)
        #tmp = torch.cat((dehaze_d1,depth_ds1), 1)
        dehaze_d2 = self.g_dehaze_e1(dehaze_d1)

        #depth_ds2 = self.downsample(depth_ds1)
        #tmp = torch.cat((dehaze_d2,depth_ds2), 1)
        dehaze_d3 = self.g_dehaze_e2(dehaze_d2)

        dehaze_t = self.g_dehaze_t(dehaze_d3)

        # decoder
        #depth_ds3 = self.downsample(depth_ds2)
        tmp = torch.cat((dehaze_t, dehaze_d3), 1)
        dehaze_us3 = self.g_dehaze_d2(tmp)

        tmp = torch.cat((dehaze_us3, dehaze_d2), 1)
        dehaze_us2 = self.g_dehaze_d1(tmp)

        tmp = torch.cat((dehaze_us2, dehaze_d1), 1)
        dehaze_us1 = self.g_dehaze_d0(tmp)

        dehaze_f = self.final(dehaze_us1)

        # pyramid enhancer
        tmp = torch.cat((dehaze_f, input), 1)
        dehaze = self.pyramid_enhancer(tmp)

        return depths, dehaze

###########################################################################
##  OmniDehaze : w/o Depth Estimation Module  ##
###########################################################################
class D_SC_Att(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=3, n_height=3, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(D_SC_Att, self).__init__()
        self.n_width = n_width
        activation = nn.ReLU(True)
        ###### DehazeGenerator #####
        ### feature extractor
        self.rwsff_0 = HeightWise_SFF_Model(ngf,height=256,reduction=4,bias=False,norm_layer=norm_layer)
        self.rwsff_1 = HeightWise_SFF_Model(ngf*2,height=256,reduction=4,bias=False,norm_layer=norm_layer)

        self.extractor_0_0 = ConELUBlock(input_nc, ngf, (3, 9), padding=(1, 4))
        self.extractor_0_1 = ConELUBlock(input_nc, ngf, (5, 11), padding=(2, 5))
        self.extractor_0_2 = ConELUBlock(input_nc, ngf, (5, 7), padding=(2, 3))
        self.extractor_0_3 = ConELUBlock(input_nc, ngf, 7, padding=3)

        self.extractor_1_0 = ConELUBlock(ngf, ngf * 2, (3, 9), padding=(1, 4))
        self.extractor_1_1 = ConELUBlock(ngf, ngf * 2, (3, 7), padding=(1, 3))
        self.extractor_1_2 = ConELUBlock(ngf, ngf * 2, (3, 5), padding=(1, 2))
        self.extractor_1_3 = ConELUBlock(ngf, ngf * 2, 5, padding=2)

        mult = 2
        g_dehaze_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e0 = nn.Sequential(*g_dehaze_e0)

        mult = 4
        g_dehaze_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e1 = nn.Sequential(*g_dehaze_e1)

        mult = 8
        g_dehaze_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
        g_dehaze_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_e2 = nn.Sequential(*g_dehaze_e2)

        ### resnet blocks
        g_dehaze_t = []
        mult = 2**(3 +1)
        for i in range(n_blocks_global):
            g_dehaze_t += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.g_dehaze_t = nn.Sequential(*g_dehaze_t)
        
        ### decoder         
        mult = 16        
        g_dehaze_d2 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d2 = nn.Sequential(*g_dehaze_d2)

        mult = 8        
        g_dehaze_d1 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d1 = nn.Sequential(*g_dehaze_d1)

        mult = 4        
        g_dehaze_d0 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
        g_dehaze_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.g_dehaze_d0 = nn.Sequential(*g_dehaze_d0)


        final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
        self.final = nn.Sequential(*final)           
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        self.pyramid_enhancer = Dehaze1()

    def forward(self, input): 
        # First filter bank
        feature_0_0 = self.extractor_0_0(input)
        feature_0_1 = self.extractor_0_1(input)
        feature_0_2 = self.extractor_0_2(input)
        feature_0_3 = self.extractor_0_3(input)
        #feature_0_cat = torch.cat((feature_0_0, feature_0_1, feature_0_2, feature_0_3), 1)

        feature_fuse_0 = self.rwsff_0(feature_0_0,feature_0_1,feature_0_2,feature_0_3)

        # Second filter bank
        feature_1_0 = self.extractor_1_0(feature_fuse_0)
        feature_1_1 = self.extractor_1_1(feature_fuse_0)
        feature_1_2 = self.extractor_1_2(feature_fuse_0)
        feature_1_3 = self.extractor_1_3(feature_fuse_0)
        #feature_1_cat = torch.cat((feature_1_0, feature_1_1, feature_1_2, feature_1_3), 1)

        feature_fuse_1 = self.rwsff_1(feature_1_0,feature_1_1,feature_1_2,feature_1_3)

        #depths = self.g_depth_coase(feature_fuse_1)
        #depth = depths[0]

        # encoder
        #tmp = torch.cat((feature_fuse_1, depth), 1)
        dehaze_d1 = self.g_dehaze_e0(feature_fuse_1)  # ds

        #depth_ds1 = self.downsample(depth)
        #tmp = torch.cat((dehaze_d1,depth_ds1), 1)
        dehaze_d2 = self.g_dehaze_e1(dehaze_d1)

        #depth_ds2 = self.downsample(depth_ds1)
        #tmp = torch.cat((dehaze_d2,depth_ds2), 1)
        dehaze_d3 = self.g_dehaze_e2(dehaze_d2)

        dehaze_t = self.g_dehaze_t(dehaze_d3)

        # decoder
        #depth_ds3 = self.downsample(depth_ds2)
        tmp = torch.cat((dehaze_t, dehaze_d3), 1)
        dehaze_us3 = self.g_dehaze_d2(tmp)

        tmp = torch.cat((dehaze_us3, dehaze_d2), 1)
        dehaze_us2 = self.g_dehaze_d1(tmp)

        tmp = torch.cat((dehaze_us2, dehaze_d1), 1)
        dehaze_us1 = self.g_dehaze_d0(tmp)

        dehaze_f = self.final(dehaze_us1)

        # pyramid enhancer
        tmp = torch.cat((dehaze_f, input), 1)
        dehaze = self.pyramid_enhancer(tmp)

        return dehaze

#######################
## Depth Estimation  ##
#######################
class CoaseDepthUNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, n_strip=2, n_blocks_global=3, 
                 n_width=3, n_height=3, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):
        super(CoaseDepthUNet, self).__init__()
        self.n_width = n_width
        
        activation_e = nn.ELU(alpha=1.0, inplace=True) 
        activation_d = nn.ELU(alpha=1.0, inplace=True)     


        ### encoder
        
        mult = 2
        g_depth_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation_e]
        g_depth_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation_e, norm_layer=norm_layer)]
        self.g_depth_e0 = nn.Sequential(*g_depth_e0)

        mult = 4
        g_depth_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation_e]
        g_depth_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation_e, norm_layer=norm_layer)]
        self.g_depth_e1 = nn.Sequential(*g_depth_e1)

        mult = 8
        g_depth_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation_e]
        g_depth_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation_e, norm_layer=norm_layer)]
        self.g_depth_e2 = nn.Sequential(*g_depth_e2)

        ### resnet blocks
        g_depth_t = []
        for i in range(1):
            g_depth_t += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation_e, norm_layer=norm_layer)]
            self.g_depth_t = nn.Sequential(*g_depth_t)
        
        ### decoder         
        mult = 16        
        g_depth_d2 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation_d]
        g_depth_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation_d, norm_layer=norm_layer)]
        self.g_depth_d2 = nn.Sequential(*g_depth_d2)

        mult = 8        
        g_depth_d1 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation_d]
        g_depth_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation_d, norm_layer=norm_layer)]
        self.g_depth_d1 = nn.Sequential(*g_depth_d1)

        final_ds = [nn.ReflectionPad2d(3), nn.Conv2d(int(ngf * mult / 2), 1, kernel_size=7, padding=0), nn.Tanh()]
        self.final_ds = nn.Sequential(*final_ds)  

        mult = 4        
        g_depth_d0 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation_d]
        g_depth_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation_d, norm_layer=norm_layer)]
        self.g_depth_d0 = nn.Sequential(*g_depth_d0)


        #final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
        final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, 1, kernel_size=7, padding=0), nn.Tanh()]
        self.final = nn.Sequential(*final)              
        
        self.downsample = nn.AvgPool2d(4, stride=2, padding=[1, 1], count_include_pad=False)
        #self.pyramid_enhancer = Dehaze1()

    def forward(self, input): 
        # encoder
        depth_d1 = self.g_depth_e0(input)
        depth_d2 = self.g_depth_e1(depth_d1)
        depth_d3 = self.g_depth_e2(depth_d2)

        depth_t = self.g_depth_t(depth_d3)
        # decoder
        tmp = torch.cat((depth_t, depth_d3), 1)
        depth_us3 = self.g_depth_d2(tmp)

        tmp = torch.cat((depth_us3, depth_d2), 1)
        depth_us2 = self.g_depth_d1(tmp)

        depth_ds = self.final_ds(depth_us2)

        tmp = torch.cat((depth_us2, depth_d1), 1)
        depth_us1 = self.g_depth_d0(tmp)

        depth = self.final(depth_us1)


        return [depth, depth_ds]





class PramidScaleAttentionNet(nn.Module):
    def __init__(self):
        super(PramidScaleAttentionNet, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(7, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze

class Dehaze1(nn.Module):
    def __init__(self):
        super(Dehaze1, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze



        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Define a resnet block
class ResnetBlock_SC(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_SC, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [SphereConv2D(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [SphereConv2D(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=2, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

