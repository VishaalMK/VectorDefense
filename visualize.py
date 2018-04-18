import matplotlib
#matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import gridspec
from decimal import *

nb_classes = 10

def AX_perturbations(fig, perturbations):
	"""
	A grid of 1x10 images representing the perturbations of the 
	generated AXs - with a single coolwarm colorbar at the right
	"""
	grid = AxesGrid(fig, 211,  # similar to subplot(211)
		    nrows_ncols=(1, 10),
		    axes_pad=0.0,
		    share_all=True,
		    label_mode="1",
		    cbar_location="right",
		    cbar_mode="single",
		    cbar_size="7%",
		    cbar_pad="3%",
		    )

	for i in range(nb_classes):
	    img = perturbations[i].reshape([28, 28])
	    im = grid[i].imshow(img, interpolation="nearest", cmap=cm.coolwarm, vmin=-1., vmax=1.)
	    grid[i].tick_params(which='both',bottom='off',left='off',labelbottom='off',labelleft='off')
	import matplotlib as mpl
	norm_ = mpl.colors.Normalize(vmin=-1.,vmax=1.)
	grid.cbar_axes[0].colorbar(im, norm=norm_)
	#grid.cbar_axes[0].set_yticklabels(['-1', '0', '1'])
	grid.cbar_axes[0].set_yticks((-1, 0, 1))

def AX_actual(fig, adv_x, top_1, confidence, ylabel):
	"""
	A grid of 1x10 images displaying the actual AXs along with 
	their predicted labels and confidences"""
	grid = AxesGrid(fig, 111,  # similar to subplot(212)
		    nrows_ncols=(1, 10),
		    axes_pad=0.0,
		    share_all=True,
		    label_mode="all"
		    )
	for i in range(nb_classes):
	    img = adv_x[i].reshape([28,28])
	    im = grid[i].imshow(img, cmap='gray')
	    grid[i].tick_params(which='both',bottom='off',left='off', labelbottom='off', labelleft='off')
	    conf = str(Decimal(str(confidence[i])).quantize(Decimal('0.01'), rounding=ROUND_DOWN))
	    #xlabel = str(top_1[i]) + " " + "(" + '{0:.2f}'.format(confidence[i]) + ')'
	    xlabel = str(top_1[i]) + " " + "(" + conf + ")"
	    grid[i].set_xlabel(xlabel, labelpad=2.0, fontsize=12)
	grid[9].yaxis.set_label_position("right")
    # ylabel should be a string
	grid[9].set_ylabel(ylabel, labelpad=14.0, fontsize=15, rotation=270)

	#grid.axes_llc.set_xticks("")

