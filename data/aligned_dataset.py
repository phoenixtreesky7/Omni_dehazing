### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset,make_sortedDataset
from PIL import Image
import matplotlib.pyplot as plt

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        #if opt.isTrain:
        #    self.B_paths,self.A_paths=make_sortedDataset('train')
        #else:
        #    dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        #    self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        #    self.A_paths = sorted(make_dataset(self.dir_A))


        # ### input A (label maps)
        #
        dir_A = '_H' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))
        #

        # ### input B (real images)
        if self.opt.isTrain:
            dir_B = '_G' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

            dir_D = '_D' if self.opt.label_nc == 0 else '_dep'
            self.dir_D = os.path.join(opt.dataroot, opt.phase + dir_D)
            self.D_paths = sorted(make_dataset(self.dir_D))


        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params, channel=3)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        D_tensor = 0
        ### input B (real images)
        if self.opt.isTrain:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params, channel=3)      
            B_tensor = transform_B(B)

            D_path = self.D_paths[index]
            D = Image.open(D_path)#.convert('RGB')
            #plt.figure("Image") # 图像窗口名称
            #plt.imshow(D)
            transform_D = get_transform(self.opt, params, channel=1)      
            D_tensor = transform_D(D)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        #print('A', A_tensor.shape)
        input_dict = {'label': A_tensor, 'inst': D_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'