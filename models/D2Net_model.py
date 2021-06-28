### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from models.OmniCriterion import *

class D2Net_model(BaseModel):
    def name(self):
        return 'D2Net_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True

        self.opt_netD = opt.netD

        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_strip, opt.n_blocks_global, opt.n_width, 
                                      opt.n_height, opt.norm, gpu_ids=self.gpu_ids)



        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            if self.opt_netD == 'coarse_refine' or self.opt_netD == 'coarse':
                self.netD_depth = networks.define_D(netD_input_nc-2, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            if self.opt_netD == 'coarse_refine' or self.opt_netD == 'refine':
                self.netD_refinedepth = networks.define_D(netD_input_nc-2, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
            
        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionMse=torch.nn.MSELoss()
            self.criterionDepth = MultiScaleL2Loss([0.535, 0.272], [0.134, 0.068])
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            if self.opt_netD == 'coarse_refine':
                self.loss_names = ['G_GAN', 'G_depth_GAN', 'G_refinedepth_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake','D_depth_real', 'D_depth_fake','D_refinedepth_real', 'D_refinedepth_fake', 'G_Dehaze_L2', 'G_Depth_L2', 'G_refineDepth_L2']
            elif self.opt_netD == 'coarse':
                self.loss_names = ['G_GAN', 'G_depth_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake','D_depth_real', 'D_depth_fake', 'G_Dehaze_L2', 'G_Depth_L2']
            elif self.opt_netD == 'refine':
                self.loss_names = ['G_GAN', 'G_refinedepth_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake','D_refinedepth_real', 'D_refinedepth_fake', 'G_Dehaze_L2', 'G_refineDepth_L2']
            elif self.opt_netD == 'dehaze':
                self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'G_Dehaze_L2']


            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_width)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
                params+=list(self.netG.dehaze.parameters())
            else:
                params = list(self.netG.parameters())

            if self.gen_features:              
                params += list(self.netE.parameters())         
            # self.optimizer_G = torch.optim.Adam([{'params':self.netG.dehaze.parameters(), 'lr':opt.lr*2},{'params':self.netG.dehaze2.parameters(), 'lr':opt.lr*2},{'params':self.netG.model.parameters(),'lr':opt.lr}], betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            if self.opt_netD == 'coarse' or opt.netD == 'coarse_refine':
                params = list(self.netD_depth.parameters())   
                self.optimizer_D_depth = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            if self.opt_netD == 'refine' or opt.netD == 'coarse_refine':
                params = list(self.netD_refinedepth.parameters()) 
                self.optimizer_D_refinedepth = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            #print('optimizer_D_refinedepth is set')




    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        # get edges from instance map
        if inst_map is not None:
            input_inst = Variable(inst_map.data.cuda())
        else:
            input_inst = None
        if not self.opt.no_instance:
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1) 
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label, input_inst, real_image,  feat_map

    def discriminate(self, input_label, test_image, use_pool=False, refine=False):
        #print(input_label.shape)
        #print(test_image.shape)
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if test_image.size(1) == 3:
            if use_pool:
                fake_query = self.fake_pool.query(input_concat)
                #print('fake_query', fake_query.shape)
                return self.netD.forward(fake_query)
            else:
                return self.netD.forward(input_concat)
        elif test_image.size(1) == 1:
            if refine:
                if use_pool:
                    fake_query = self.fake_pool.query(input_concat)
                    return self.netD_refinedepth.forward(fake_query)
                else:
                    return self.netD_refinedepth.forward(input_concat)
            else:
                if use_pool:
                    fake_query = self.fake_pool.query(input_concat)
                    return self.netD_depth.forward(fake_query)
                else:
                    return self.netD_depth.forward(input_concat)

        

    def forward(self, label, inst, image, feat, infer=False):
        # Encode Inputs
        input_label, depth_real_image, real_image, feat_map = self.encode_input(label, inst, image, feat) 
        #print('label',input_label.shape, 'depth',depth_real_image.shape,'real',real_image.shape) 
        #depth_real_image = (depth_real_image[0]+depth_real_image[1]+depth_real_image[2])/3

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)                     
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label


        if self.opt_netD == 'coarse_refine':
            depth_image,dehazed_image, depth_refine = self.netG.forward(input_concat)
        elif self.opt_netD == 'coarse':
            depth_image,dehazed_image = self.netG.forward(input_concat)
            depth_refine = depth_image
        elif self.opt_netD == 'refine':
            dehazed_image, depth_refine = self.netG.forward(input_concat)
            depth_image = depth_refine
        elif self.opt_netD == 'dehaze':
            dehazed_image = self.netG.forward(input_concat)


        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, dehazed_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, dehazed_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)  

        # depth ###################
        if self.opt_netD == 'coarse' or self.opt_netD == 'coarse_refine':
            pred_fake_pool_depth = self.discriminate(input_label, depth_image[0], use_pool=True)
            loss_D_fake_depth = self.criterionGAN(pred_fake_pool_depth, False)        

            # Real Detection and Loss        
            pred_real_depth = self.discriminate(input_label, depth_real_image)
            loss_D_real_depth = self.criterionGAN(pred_real_depth, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake_depth = self.netD_depth.forward(torch.cat((input_label, depth_image[0]), dim=1))
            loss_G_GAN_depth = self.criterionGAN(pred_fake_depth, True)

        #################################
        if self.opt_netD == 'refine' or self.opt_netD == 'coarse_refine':

            pred_fake_pool_refinedepth = self.discriminate(input_label, depth_refine[0], use_pool=True,refine=True)
            loss_D_fake_refinedepth = self.criterionGAN(pred_fake_pool_refinedepth, False)        

            # Real Detection and Loss        
            pred_real_refinedepth = self.discriminate(input_label, depth_real_image,refine=True)
            loss_D_real_refinedepth = self.criterionGAN(pred_real_refinedepth, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake_refinedepth = self.netD_refinedepth.forward(torch.cat((input_label, depth_refine[0]), dim=1))
            loss_G_GAN_refinedepth = self.criterionGAN(pred_fake_refinedepth, True)          
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        pred_fake = self.netD.forward(torch.cat((input_label, dehazed_image), dim=1))
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(dehazed_image, real_image) * self.opt.lambda_feat
        loss_Dehaze_L2 = self.criterionMse(dehazed_image, real_image)

        if self.opt_netD == 'coarse_refine':
            loss_Depth_L2 = self.criterionDepth(depth_image,depth_real_image,F.interpolate(depth_real_image,scale_factor=0.5))
            loss_refineDepth_L2 = self.criterionDepth(depth_refine,depth_real_image,F.interpolate(depth_real_image,scale_factor=0.5))
            coarse_depth = depth_image[0]
            refine_depth = depth_refine[0]
            losses = [loss_G_GAN, loss_G_GAN_depth, loss_G_GAN_refinedepth, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_D_real_depth, loss_D_fake_depth, loss_D_real_refinedepth, loss_D_fake_refinedepth, loss_Dehaze_L2, loss_Depth_L2, loss_refineDepth_L2]
        elif self.opt_netD == 'coarse':
            loss_Depth_L2 = self.criterionDepth(depth_image,depth_real_image,F.interpolate(depth_real_image,scale_factor=0.5))
            coarse_depth = depth_image[0]
            refine_depth = depth_refine[0]
            losses = [loss_G_GAN, loss_G_GAN_depth, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_D_real_depth, loss_D_fake_depth,  loss_Dehaze_L2, loss_Depth_L2]
        elif self.opt_netD == 'refine':
            loss_refineDepth_L2 = self.criterionDepth(depth_refine,depth_real_image,F.interpolate(depth_real_image,scale_factor=0.5))
            coarse_depth = depth_image[0]
            refine_depth = depth_refine[0]
            losses = [loss_G_GAN, loss_G_GAN_refinedepth, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_D_real_refinedepth, loss_D_fake_refinedepth, loss_Dehaze_L2, loss_refineDepth_L2]
        elif self.opt_netD == 'dehaze':
            losses = [loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_Dehaze_L2]
            coarse_depth = dehazed_image[0]
            refine_depth = dehazed_image[0]

        

        
        # Only return the fake_B image if necessary to save BW
        return [ losses, coarse_depth, refine_depth, dehazed_image]
    # self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake','G_L2']
    def inference(self, label):
        # Encode Inputs        
        input_label, _, _, _ = self.encode_input(Variable(label), infer=True)

        # Fake Generation
        if self.use_features:       
            # sample clusters from precomputed features             
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label                
        coarse_depth, dehazed_image = self.netG.forward(input_concat)
        return coarse_depth[0], dehazed_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])                   
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == i).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k] 
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        if opt.netD == 'coarse_refine' or opt.netD == 'refine':
            for param_group in self.optimizer_D_refinedepth.param_groups:
                param_group['lr'] = lr
        if opt.netD == 'coarse_refine' or opt.netD == 'coarse':
            for param_group in self.optimizer_D_depth.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
