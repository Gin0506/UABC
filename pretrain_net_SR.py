import torch
import torch.optim
import torch.nn.functional as F

import cv2
import os.path
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import utils.utils_image as util
import utils.utils_deblur as util_deblur
import utils.utils_psf as util_psf
from models.uabcnet import UABCNet as net
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_kernels(kernel_path):
	kernels = []
	kernel_files = glob.glob(os.path.join(kernel_path,'*.npz'))
	kernel_files.sort()
	for kf in kernel_files:
		PSF_grid = np.load(kf)['PSF']
		PSF_grid = util_psf.normalize_PSF(PSF_grid)
		kernels.append(PSF_grid)
	return kernels

def draw_random_kernel(kernels,patch_num):
	# n = len(kernels)
	# i = np.random.randint(2*n)
	# if i<0:
	# 	psf = kernels[i]
	# else:
	# 	psf = gaussian_kernel_map(patch_num)
	# return psf
	psf = np.loadtxt('E:\Personal\Megapixel\psf\psf_phase_crop.txt').astype(np.float32)
	psf = psf[...,None].repeat(3,axis=-1)
	psf = psf[None,...].repeat(2,axis=0)
	psf = psf[None, ...].repeat(2, axis=0)
	psf = util_psf.normalize_PSF(psf)

	return psf

def gaussian_kernel_map(patch_num):
	PSF = np.zeros((patch_num[0],patch_num[1],25,25,3))
	for w_ in range(patch_num[0]):
		for h_ in range(patch_num[1]):
			PSF[w_,h_,...,0] = util_deblur.gen_kernel()
			PSF[w_,h_,...,1] = util_deblur.gen_kernel()
			PSF[w_,h_,...,2] = util_deblur.gen_kernel()
	return PSF


def draw_training_pairGPU(image_H, psf,sf, patch_num, patch_size,img_L=None):
	w, h = image_H.shape[:2]
	gx, gy = psf.shape[:2]
	px_start = np.random.randint(0, gx-patch_num[0]+1)
	py_start = np.random.randint(0, gy-patch_num[1]+1)

	psf_patch = psf[px_start:px_start+patch_num[0],
					py_start:py_start+patch_num[1], ...]
	patch_size_H = [patch_size[0]*sf,patch_size[1]*sf]

	# generate image_L on-the-fly
	conv_expand = psf.shape[-1]//2
	x_start = np.random.randint(
		0, w-patch_size_H[0]*patch_num[0]-conv_expand*2+1)
	y_start = np.random.randint(
		0, h-patch_size_H[1]*patch_num[1]-conv_expand*2+1)
	patch_H = image_H[x_start:x_start+patch_size_H[0]*patch_num[0]+conv_expand*2,
					  y_start:y_start+patch_size_H[1]*patch_num[1]+conv_expand*2]
	patch_H = util.uint2tensor4(patch_H)
	patch_H = patch_H.to(psf_patch.device)

	patch_L = util_deblur.blockConv2dGPU(patch_H, psf_patch, conv_expand)

	patch_H = patch_H[0, ..., conv_expand:-conv_expand, conv_expand:-conv_expand]
	patch_L = patch_L[:,::sf, ::sf]
	return patch_L, patch_H, psf_patch


def draw_training_pair(image_H,psf,sf,patch_num,patch_size,image_L=None):
	w,h = image_H.shape[:2]
	gx,gy = psf.shape[:2]
	px_start = np.random.randint(0,gx-patch_num[0]+1)
	py_start = np.random.randint(0,gy-patch_num[1]+1)

	psf_patch = psf[px_start:px_start+patch_num[0],py_start:py_start+patch_num[1]]
	patch_size_H = [patch_size[0]*sf,patch_size[1]*sf]

	if image_L is None:
		#generate image_L on-the-fly
		conv_expand = psf.shape[2]//2
		x_start = np.random.randint(0,w-patch_size_H[0]*patch_num[0]-conv_expand*2+1)
		y_start = np.random.randint(0,h-patch_size_H[1]*patch_num[1]-conv_expand*2+1)
		patch_H = image_H[x_start:x_start+patch_size_H[0]*patch_num[0]+conv_expand*2,\
		y_start:y_start+patch_size_H[1]*patch_num[1]+conv_expand*2]
		patch_L = util_deblur.blockConv2d(patch_H,psf_patch,conv_expand)

		patch_H = patch_H[conv_expand:-conv_expand,conv_expand:-conv_expand]
		patch_L = patch_L[::sf,::sf]

	else:
		x_start = px_start * patch_size_H[0]
		y_start = py_start * patch_size_H[1]
		patch_H = image_H[x_start:x_start+patch_size_H[0]*patch_num[0],\
			y_start:y_start+patch_size_H[1]*patch_num[1]]
		x_start = px_start * patch_size[0]
		y_start = py_start * patch_size[1]
		patch_L = image_L[x_start:x_start+patch_size[0]*patch_num[0],\
			y_start:y_start+patch_size[1]*patch_num[1]]

	return patch_L,patch_H,psf_patch

def main():
	#0. global config
	#scale factor
	sf = 4	
	stage = 8

	batch_size = 3
	patch_size = [32,32]
	patch_num = [2,2]

	#1. local PSF
	#shape: gx,gy,kw,kw,3
	all_PSFs = load_kernels('./data')

	#2. local model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=7, out_nc=3, nc=[64, 128, 256, 512],
					nb=2,sf=sf, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	#model.proj.load_state_dict(torch.load('./data/usrnet_pretrain.pth'),strict=True)
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	#positional lambda, mu for HQS, set as free trainable parameters here.
	ab_buffer = np.ones((1,1,2*stage,3),dtype=np.float32)
	ab_buffer[...,1::2,:] = 0.5
	ab_buffer[...,::2, :] = np.log(np.exp(1e-3)-1)

	ab = torch.tensor(ab_buffer,device=device,requires_grad=True)

	params = []
	all_PSNR = []
	params += [{"params":[ab],"lr":0.0005}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":0.0001}]
	optimizer = torch.optim.Adam(params,lr=0.0001,betas=(0.9,0.999))
	
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10000000,gamma=0.9)

	#3.load training data
	imgs_H = glob.glob('./DIV2K_train/*.png',recursive=True)
	imgs_L = glob.glob('./DIV2K_lr/*.png', recursive=True)
	imgs_H.sort()
	imgs_L.sort()

	global_iter = 0
	N_maxiter = 80000

	#def get_train_pairs()

	for i in range(N_maxiter):

		t0 = time.time()
		#draw random image.
		loss = 0

		for _ in range(batch_size):

			img_idx = np.random.randint(len(imgs_H))

			img_H = cv2.imread(imgs_H[img_idx])

			#img2 = imgs_L[img_idx]
			#img_L = cv2.imread(img2)
			#draw random patch from image
			#a. without img_L

			#draw random kernel
			PSF_grid = draw_random_kernel(all_PSFs,patch_num)
			PSF_grid = PSF_grid.transpose(0,1,4,2,3)
			k_GPU = torch.tensor(PSF_grid).to(device).float()


			x,x_gt,patch_psf = draw_training_pairGPU(img_H,k_GPU,sf,patch_num,patch_size)
			#b.	with img_L
			#patch_L, patch_H, patch_psf,px_start, py_start,block_expand = draw_training_pair(img_H, PSF_grid, sf, patch_num, patch_size, img_L)
			t_data = time.time()-t0

			x = x[None,...]
			x_gt = x_gt[None,...]

			k_local = []
			for h_ in range(patch_num[1]):
				for w_ in range(patch_num[0]):
					local_psf = patch_psf[w_, h_]
					k_local.append(local_psf)
			k = torch.stack(k_local, dim=0)

			[x,x_gt,k] = [el.to(device) for el in [x,x_gt,k]]

			ab_patch = F.softplus(ab).expand(patch_num[0],patch_num[1],2*stage,3)
			ab_patch_v = []
			for h_ in range(patch_num[1]):
				for w_ in range(patch_num[0]):
					ab_patch_v.append(ab_patch[w_:w_+1,h_])
			ab_patch_v = torch.cat(ab_patch_v,dim=0)
			# x_E = model.forward_patchwise_SR(x, k, ab_patch_v, patch_num,
			# 												  [patch_size[0], patch_size[1]], sf)
			x_E,outputs,x_init = model.forward_patchwise_SR(x,k,ab_patch_v,patch_num,[patch_size[0],patch_size[1]],sf)

			loss += F.l1_loss(x_E,x_gt)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		t_iter = time.time() - t0 - t_data

		print('[iter:{}] loss:{:.4f}, data_time:{:.2f}s, net_time:{:.2f}s'.format(global_iter+1,loss.item(),t_data,t_iter))

		patch_H = util.tensor2uint(x_gt)
		patch_L = util.tensor2uint(x)
		patch_E = util.tensor2uint(x_E)

		patch_L = cv2.resize(patch_L,dsize=None,fx=sf,fy=sf,interpolation=cv2.INTER_NEAREST)
		#patch_L = patch_L[block_expand*sf:-block_expand*sf,block_expand*sf:-block_expand*sf]

		psnr = cv2.PSNR(patch_E, patch_H)
		all_PSNR.append(psnr)
		show = np.hstack((patch_H,patch_L,patch_E))
		cv2.imshow('H,L,E',show)

		# outputs = [util.tensor2uint(elem) for elem in outputs]
		# x_init = util.tensor2uint(x_init)
		# z_all = np.hstack(outputs[::2])
		# x_all = np.hstack(outputs[1::2])
		# cv2.imshow('z_all', z_all)
		# cv2.imshow('x_all', x_all)

		key = cv2.waitKey(1)
		global_iter+= 1

		if (i - 79000) > 0:
			cv2.imwrite(os.path.join('./result', 'test1', 'hstack-{:04d}.png'.format(i + 1)), show)

		if key==ord('q'):
			break
		if key==ord('s'):
			pass
			ab_numpy = ab.detach().cpu().numpy().flatten()
			np.savetxt('./logs/ab_pretrain.txt',ab_numpy)
			torch.save(model.state_dict(),'./logs/uabcnet.pth')

	torch.save(model.state_dict(),'./logs/uabcnet_final.pth')
	ab_numpy = ab.detach().cpu().numpy().flatten()
	np.savetxt('./logs/ab_pretrain.txt', ab_numpy)
	np.savetxt(os.path.join('./result', 'test1', 'psnr.txt'), all_PSNR)

if __name__ == '__main__':

	main()
