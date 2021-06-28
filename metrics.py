import os
from PIL import Image
import pytorch_ssim
import numpy as np
dir_img="D:/dzhao/dehazing_360/360D2Net/Pix2PixHD_modify/results/p2hed_sc/test_20/images/"
dir_gt="D:/dzhao/dehazing_360/360hazy_dataset/test/test_G/"

list_img = os.listdir(dir_img) #获取目录下所有图片名
list_gt = os.listdir(dir_gt)

list_all = [list_img, list_gt]
print(len(list_gt))

for num in range(len(list_img)):
	gt = np.asarray(Image.open(dir_gt+list_gt[num]))
	image = np.asarray(Image.open(dir_img+list_img[num]))

	ssim_my = pytorch_ssim.ssim(c, image)
	avg_ssim_my += ssim_my
	mse_my = criterion(gt, image)
	psnr_my = 10 * log10(1 / mse_my)
	avg_psnr_my += psnr_my_my

print('avg-ssim', avg_ssim_my/1413)
print('avg-psnr', avg_psnr_my/1413)
