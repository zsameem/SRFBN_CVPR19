import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import glob
import os

def extract_noise_patches(img_dir, save_dir, patch_size=64):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    files = glob.glob(img_dir+'/*')
    for f_count, file_path in enumerate(files):
        print(f_count, file_path)
        img = cv2.imread(file_path)
        smoothed = np.zeros((img.shape))
        for ch in range(img.shape[2]):
            smoothed[:,:,ch] = gaussian_filter(img[:,:,ch], 10)
        for i in range(0, smoothed.shape[0]-patch_size, patch_size):
            for j in range(0, smoothed.shape[1]-patch_size, patch_size):
                patch = smoothed[i:i+patch_size, j:j+patch_size, :]
                if np.std(patch) < 3:
                    noise = img[i:i+patch_size, j:j+patch_size, :] - patch
#                     noise/np.max(noise)
#                     noise = np.clip(noise, 0, 1)
                    noise = np.clip(noise, 0, 255)
                    noise = noise.astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, 'patch_{:05d}_{:06d}.png'.format(f_count, i+64*j)), noise)
#                     plt.imsave('./noise_patches/patch_{:06d}.png'.format(i+64*j), noise)
#                     print(i, j, np.std(patch))
        if (f_count > 300):
            break

if __name__=='__main__':
    img_dir = '/home/samim/Desktop/ms/real-world-sr/datasets/original_images/train/iphone'
    save_dir = './datasets/dped_noise_patches/iphone'
    extract_noise_patches (img_dir, save_dir)