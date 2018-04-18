import os
import random
import PIL
import PIL.Image
import numpy as np
import sys

from cleverhans.utils_mnist import data_mnist

SAMPLES = 60000
DIM = 4  # tentative
OUTPUT_FILE = 'data/mnist_quilt_db.npy'

def main():
	idx = 0
	assert SAMPLES % 1000 == 0
	
	db = np.zeros((SAMPLES, DIM, DIM, 1), dtype=np.float32)

        X_train, Y_train, X_test, Y_test = data_mnist()
        for f in random.sample(np.arange(len(X_train)), SAMPLES):
		img = X_train[f]
		h, w, _ = img.shape
		h_start = random.randint(0, h - DIM)
		w_start = random.randint(0, w - DIM)
		crop = img[h_start:h_start+DIM, w_start:w_start+DIM, :]
		db[idx, :, :, :] = crop
		idx += 1

		if idx % 100 == 0:
			print('%.2f%% done' % (100 * (float(idx) / SAMPLES)))

	np.save(OUTPUT_FILE, db)

if __name__=='__main__':
	main()
