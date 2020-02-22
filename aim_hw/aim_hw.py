import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import itertools

if __name__ ==  "__main__":
    # read image
    img = plt.imread( 'Lena.png' )
    
    # show original
    plt.imshow( img )

    imcopy = img.copy()
    
    # modify image
    for i in range(0, 512):
        for j in range(0, 512):
            imcopy[i, j] = (1 - img[i, j])

    # save new image
    plt.imsave('modified.png', imcopy)