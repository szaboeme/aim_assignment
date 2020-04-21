import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import os
import itertools
import math

# evaluate 1D Gaussian in point p
def evalGauss1(p, sigma):
    # p is already squared
    return (1.0 / (sigma * math.sqrt(2.0 * np.pi))) * (math.exp(-( p ) / (2.0 * sigma * sigma)))

# evaluate 2D Gaussian in point p
def evalGauss2(p, sigma):
    # p is already squared
    return (1.0 / (sigma * sigma * (2.0 * np.pi))) * (math.exp(-( p ) / (2.0 * sigma * sigma)))


# apply bilateral filter
def apply(img, pad, fs, sigmag, sigmab):
    ims = img.shape[0]

    for i in range(pad, ims - pad):
        for j in range(pad, ims - pad):
            r = 0
            W = 0
            for k in range(fs):
                for k2 in range(fs):
                    # in b goes the actual image difference
                    b = evalGauss1( (img[i, j] - img[i-pad+k, j-pad+k2])**2, sigmab )
                    # in G go the coordinate differences
                    G = evalGauss2( (-pad + k)**2 + (-pad + k2)**2, sigmag )
                    weight = b * G
                    r += img[i - pad + k, j - pad + k2] * weight
                    W += weight
            img[i, j] = r / W

    return img


if __name__ == "__main__":
    print("Welcome to PPAFBF (Primitive Python Application For Bilateral Filtering)!")
    imgs = os.listdir('images/')
    print("\nPlease choose an image from the list to be loaded:\n")
    for i in imgs:
        print(i)
    #img_name = input("\nSelected image name (without '.png'): ")
    img_name = "noisy"
    file_name = "images/" + img_name + ".png"

    # check if existing image
    if not os.path.isfile(file_name):
        try:
            sys.exit(0)
        finally:
            print("Invalid file name! Exiting.")

    # read chosen image
    img = plt.imread(file_name)

    # if more than 2 image dimensions
    if (img.ndim > 2):
        img = img[:,:,0] # take one channel only

    sigg = float(input("\nSigma value for 2D kernel (G): "))
    sigb = float(input("\nSigma value for 1D kernel (b): "))
    fsize = int(input("\nFilter size: "))
    #sigg = 10.0 
    #sigb = 100.0 
    #fsize = 15

    # create filter
    #filter = gauss2d(sig, fsize)

    imsize = img.shape[0]
    pad = int(fsize/2)

    # padding in one direction
    leftpadding = []
    rightpadding = []
    for k in range(pad):
        leftpadding.append(img[:, 0])
        rightpadding.append(img[:, imsize-1])

    leftpadding = np.reshape(np.transpose(np.array(leftpadding)), (imsize, pad))
    rightpadding = np.reshape(np.transpose(np.array(rightpadding)), (imsize, pad))
    padded = np.hstack([leftpadding, img, rightpadding])
    result = padded.copy()

    leftpadding = []
    rightpadding = []
    for k in range(pad):
        leftpadding.append(result[0, :])
        rightpadding.append(result[imsize-1, :])

    leftpadding = np.reshape(np.array(leftpadding), (pad, imsize+2*pad))
    rightpadding = np.reshape(np.array(rightpadding), (pad, imsize+2*pad))

    padded = np.vstack([leftpadding, result, rightpadding])
    result2 = padded.copy()
    #sigg = 3.0
    #sigb = 30.0
    result2 = apply(result2, pad, fsize, sigg, sigb)
    result2 = result2[pad:imsize + pad, pad:imsize + pad]
    
    #plt.subplot(131), plt.imshow(result2, cmap='gray'), plt.title("$/sigma_G$=" + str(sigg) + " $/sigma_b$=" + str(sigb)), plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("original"), plt.axis('off')
    plt.subplot(122), plt.imshow(result2, cmap='gray'), plt.title("filtered"), plt.axis('off')
    #plt.subplot(331), plt.imshow(result2, cmap='gray'), plt.axis('off')
    
    res2_scaled = 255.0 * result2
    plt.imsave('results/' + img_name + '_sig' + str(sigg) + '_sigb' + str(sigb) + '_f' + str(fsize) + '.png', res2_scaled, vmin=0, vmax=255, cmap='gray')
    #plt.imsave('results/' + img_name + '_sigg_sigb.png', res2_scaled, vmin=0, vmax=255, cmap='gray')
    
    #result2 = apply(padded.copy(), pad, fsize, 1.0, 100.0)
    #plt.subplot(332), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    #result2 = apply(padded.copy(), pad, fsize, 1.0, 300.0)
    #plt.subplot(333), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    #second row
    #result2 = apply(padded.copy(), pad, fsize, 3.0, 30.0)
    #plt.subplot(334), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    #result2 = apply(padded.copy(), pad, fsize, 3.0, 100.0)
    #plt.subplot(335), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    #result2 = apply(padded.copy(), pad, fsize, 3.0, 300.0)
    #plt.subplot(336), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    # third row
    #result2 = apply(padded.copy(), pad, fsize, 10.0, 30.0)
    #plt.subplot(337), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    #result2 = apply(padded.copy(), pad, fsize, 10.0, 100.0)
    #plt.subplot(338), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    #result2 = apply(padded.copy(), pad, fsize, 10.0, 300.0)
    #plt.subplot(339), plt.imshow(result2[pad:imsize+pad, pad:imsize+pad], cmap='gray'), plt.axis('off')

    print("Done! See the results/ folder for the saved image.")
    plt.savefig('results/'+img_name+'_full_'+str(sigg)+'_'+str(sigb)+'_f'+str(fsize)+'.png')

    # show figure
    plt.show()