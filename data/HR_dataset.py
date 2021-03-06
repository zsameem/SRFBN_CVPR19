import torch.utils.data as data
import numpy as np
import random

from data import common
from scipy import misc
from scipy.signal import convolve2d


class HRDataset(data.Dataset):
    '''
    Read HR images in train phase and generate the LR image on the fly from the 
    HR image. The LR image is generated by:
    1) Convolution of HR image with a blur kernel
    2) Subsampling
    3) Addtion of noise
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(HRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None
        if self.train:
            self.lr_size = self.opt['LR_size']
            self.hr_size = self.lr_size * self.scale
            # change the length of train dataset (influence the number of iterations in each epoch)
            self.repeat = 2
        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])
        assert self.paths_HR, '[Error] HR paths are empty.'
        self.filters_path = opt['blur_kernel_path']
        self.filter_bank = common.get_filters(self.filters_path)
        assert (len(self.filter_bank) > 0)
        self.n_filters = len(self.filter_bank)
        
        if opt['noise_patch_path']:
            self.paths_noise_patches = common.get_image_paths(self.opt['data_type'], self.opt['noise_patch_path'])
            self.n_noise_patches = len(self.paths_noise_patches)
            assert (self.n_noise_patches > 0)
            print("Number of noise patches = {}".format(self.n_noise_patches))


    def __getitem__(self, idx):
        hr, hr_path = self._load_file(idx)
        lr = None
        lr_path = None
        # Generate LR image from the HR image on the fly.
        hr, lr = self.generate_lr_hr(hr, idx)
        # LR_PATH key is not really used anywhere so its ok to set it to an
        # arbitrary value.
        lr_path = 'none'
        # if self.train:
        #     lr, hr = self._get_patch(lr, hr)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.opt['rgb_range'])
        return {'LR': lr_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}

    
    def extract_hrpatch(self, hr, patch_h, patch_w):
        ih, iw = hr.shape[:2]
        ix = random.randrange(0, iw - patch_h + 1)
        iy = random.randrange(0, ih - patch_h + 1)
        patch = hr[iy:iy + patch_h, ix:ix + patch_w, :]
        return patch


    def get_noise_patch(self, patch_shape):
        idx = random.randrange(0, self.n_noise_patches)
        noise_patch_path = self.paths_noise_patches[idx]
        noise_patch = common.read_img(noise_patch_path, self.opt['data_type'])
        return noise_patch

        
    def generate_lr_hr(self, hr, idx):
        if self.train:
            # pick a blur kernel randomly
            k_idx = random.randrange(0,self.n_filters)
            kernel = self.filter_bank[k_idx]
            if len(kernel.shape) > 2:
                assert(False, "cannot handle 3d kernels")
            [k_h, k_w] = kernel.shape[:2]
            [h, w, c] = hr.shape[:3]
            
            # After convolution with the  blur kernel the patch size should be equal to self.hr_size
            # The valid sizez before convolution are given below
            patch_h = self.hr_size + k_h - 1
            patch_w = self.hr_size + k_w - 1

            # randomly extract a patch of this size from the hr image.
            hr_patch = self.extract_hrpatch(hr, patch_h, patch_w)

            blurred_stack = []
            # blur the patch using the kernel
            for ch in range(c):
                blurred_stack.append(convolve2d(hr_patch[:,:,ch], kernel, mode='valid'))
            blurred = np.stack(blurred_stack, axis=2)
            assert(blurred.shape[0]==self.hr_size, "Blurred shape = {}, valid hr shape = {}".format(blurred.shape, self.hr_size))
            lr = blurred[::self.scale, ::self.scale, :]
            assert(lr.shape[0]==self.lr_size, "lr shape = {}, valid shape = {}".format(lr.shape, self.lr_size))
            valid_hr_patch = hr_patch[int(k_h/2):-int(k_h/2), int(k_w/2):-int(k_w/2), :]
            
            noise_patch = self.get_noise_patch(lr.shape)
            noise_scaling = np.random.random()*3 + 1
            lr += noise_scaling*noise_patch
            lr = np.clip(lr, 0, 255)
            lr, valid_hr_patch = common.augment([lr, valid_hr_patch])
            # lr = misc.imresize(hr, 1/self.scale, interp='bicubic')
            # Add noise.
            # noise_level = random.random()*3
            # noise_patch = np.random.normal(0, noise_level, (self.lr_size, self.lr_size))
            # lr = lr + noise_patch[:,:,None]
            # lr = np.clip(lr, 0, 255)
            return valid_hr_patch, lr
        else:
            k_idx = idx % self.n_filters
            kernel = self.filter_bank[k_idx]
            [k_h, k_w] = kernel.shape[:2]
            [h, w, c] = hr.shape[:3]
            blurred_stack = []
            # blur the patch using the kernel
            for ch in range(c):
                blurred_stack.append(convolve2d(hr[:,:,ch], kernel, mode='valid'))
            blurred = np.stack(blurred_stack, axis=2)
            lr = blurred[::self.scale, ::self.scale, :]
            valid_hr = hr[int(k_h/2):-int(k_h/2), int(k_w/2):-int(k_w/2), :]
            noise_level = random.random()*12
            noise_patch = np.random.normal(0, noise_level, (lr.shape[0], lr.shape[1]))
            lr = lr + noise_patch[:,:,None]
            lr = np.clip(lr, 0, 255)
            return valid_hr, lr


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_HR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        # lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        # lr = common.read_img(lr_path, self.opt['data_type'])
        hr = common.read_img(hr_path, self.opt['data_type'])

        return hr, hr_path

    # def _get_patch(self, hr):
        

    def _get_patch(self, lr, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = common.get_patch(
            lr, hr, LR_size, self.scale)
        lr, hr = common.augment([lr, hr])
        lr = common.add_noise(lr, self.opt['noise'])

        return lr, hr


