import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import itertools

def nonlincontrast(value, alpha):
    gamma = 1 / (1 - alpha)
    if value < 0.5:
        newvalue = 0.5 * pow(2*value, gamma)
    else:
        newvalue = 1 - (0.5 * pow(2 - 2*value, gamma))

    return newvalue

if __name__ ==  "__main__":
    # read image
    img = plt.imread( 'Lena.png' )
    
    # show original
    plt.imshow( img )

    neg = img.copy()
    tresh = img.copy()
    bri = img.copy()
    cont = img.copy()
    gamma = img.copy()
    nl_cont = img.copy()

    # modify image
    for i in range(0, 256):
        for j in range(0, 256):
            # negatice image
            neg[i, j] = (1 - img[i, j])
            # threshold = 0.5
            if img[i, j, 0] < 0.5:
                tresh[i, j, :] = 0
            else:
                tresh[i, j, :] = 1
            # brightness +0.2
            v = img[i, j, 0] + 0.2
            bri[i, j, :] = max(min(v, 1), 0)
            # contrast 1.25
            v = img[i, j, 0] * 1.25
            cont[i, j, :] = max(min(v, 1), 0)
            # gamma 2.0
            gamma[i, j] = pow(img[i, j], 0.5)
            # non-linear contrast
            nl_cont[i, j, :] = nonlincontrast(img[i, j, 0], 0.5)


    # save new images
    plt.imsave('negative.png', neg)
    plt.imsave('treshold.png', tresh)
    plt.imsave('brightness.png', bri)
    plt.imsave('contrast.png', cont)
    plt.imsave('gamma.png', gamma)
    plt.imsave('nl_contrast.png', nl_cont)