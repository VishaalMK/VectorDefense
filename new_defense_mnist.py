# Script to take in a numpy array, encode it to JPEG,
# vectorize->rasterize, decode png to numpy array

import tensorflow as tf
import numpy as np
import scipy.misc
import subprocess

import matplotlib
matplotlib.use('Agg')

# path to vectorization bash script
sdir = '/home'

def defend_vectorize(sess, x):
    # Get the number of images in the array
    nb_imgs = x.shape[0]
    
    for i in range(nb_imgs):
        img = x[i:i+1][0].reshape(28,28)
        # Encode the input numpy array into PNG
        out = "adv_images/" + '{0:05d}'.format(i) + ".png"
        scipy.misc.imsave(out, img)

    # Vectorize + Rasterize
    cmd = subprocess.call(sdir + 'mnist_defend.sh')

    # Decode the JPEG into numpy array
    # Read the vectorized->rasterized adversarial images
    import matplotlib.image as mpimg
    rasters = []
    for i in range(nb_imgs):
        filename = 'rasterize/' + str(i).zfill(5) + '.png'
        img = mpimg.imread(filename)
        img_gray = (1 - img[:,:,-1]).reshape(28,28,1)
        rasters.append(img_gray)

    adv_pur = np.asarray(rasters)
        
    return adv_pur
