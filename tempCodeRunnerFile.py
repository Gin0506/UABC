


def draw_training_pair(image_H,psf,sf,patch_num,patch_size,image_L=None):
	w,h = image_H.shape[:2]
	gx,gy = psf.shape[:2]
	px_start = np.random.randint(0,gx-patch_num[0]+1)