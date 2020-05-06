import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import os
import itertools
import math

# evaluate a Gaussian in point p
def evalGauss(p, sigma):
    # p is already squared; without normalization
    return (math.exp(-( p ) / (2.0 * sigma * sigma)))

# apply bilateral filter
def apply(img, pad, fs, sigmag, sigmab):
    ims = img.shape[0]
    res = img.copy()

    for i in range(pad, ims - pad):
        for j in range(pad, ims - pad):
            #r = [0,0,0]
            r = 0
            W = 0
            for k in range(fs):
                for k2 in range(fs):
                    # in b goes the actual image difference
                    #b = evalGauss( (np.linalg.norm(img[i, j] - img[i-pad+k, j-pad+k2]))**2, sigmab )
                    b = evalGauss( ((img[i, j] - img[i-pad+k, j-pad+k2]))**2, sigmab )
                    # in G go the coordinate differences
                    G = evalGauss( (-pad + k)**2 + (-pad + k2)**2, sigmag )
                    weight = b * G
                    r += img[i - pad + k, j - pad + k2] * weight
                    W += weight
            res[i, j] = r / W 

    return res

# normalize image to 0..1 range
def imnorm(img):
    min = np.amin(img)
    max = np.amax(img)
    return (img - min) * (1.0 / (max - min))

# returns a grayscale image from rgb (1 channel only)
def rgb2gray(img):
    wr = 0.3
    wg = 0.59
    wb = 0.11
    return (wr*img[:,:,0] + wg*img[:,:,1] + wb*img[:,:,2])


# computes normalized gradient image from img 
# ax=0 .. x, ax=1 .. y
def gradient(img, ax):
    sizex, sizey, channels = img.shape
    imcopy = img.copy()
    imcopy2 = img.copy()
    if ax == 0:
        padding = np.full((1,sizey,channels), 0)
        imcopy = imcopy[:-1, :, :]
        imcopy = np.vstack([padding, imcopy])
        imcopy2 = imcopy2[1:, :, :]
        imcopy2 = np.vstack([imcopy2, padding])
        imcopy = rgb2gray(imcopy)
        imcopy2 = rgb2gray(imcopy2)
    elif ax == 1:
        padding = np.full((sizex,1,channels), 0)
        imcopy = imcopy[:, :-1, :]
        imcopy = np.hstack([padding, imcopy])
        imcopy2 = imcopy2[:, 1:, :]
        imcopy2 = np.hstack([imcopy2, padding])
        imcopy = rgb2gray(imcopy)
        imcopy2 = rgb2gray(imcopy2)

    return imnorm(imcopy2[:,:] - imcopy[:,:])

if __name__ == "__main__":
    print("Welcome to PPAFIE (Primitive Python Application For Image Editing)!")
    imgs = os.listdir('images/')
    print("\nPlease choose an image from the list to be loaded:\n")
    for i in imgs:
        print(i)
    #img_name = input("\nSelected image name (without '.png'): ")
    img_name = "tunder"
    file_name = "images/" + img_name + ".png"
    # check if existing image
    if not os.path.isfile(file_name):
        try:
            sys.exit(0)
        finally:
            print("Invalid file name! Exiting.")

    #img_name2 = input("\nSelect another image (of the same size): ")
    img_name2 = "tover"
    file_name2 = "images/" + img_name2 + ".png"
    # check if existing image
    if not os.path.isfile(file_name2):
        try:
            sys.exit(0)
        finally:
            print("Invalid file name! Exiting.")
    
    # read chosen image
    img1 = plt.imread(file_name)
    img2 = plt.imread(file_name2)

    sizex, sizey, channels = img1.shape
    
    # for both images:
    # compute image gradient in x way
    gradX1 = gradient(img1, 0)
    # check
    plt.imsave('results/gradx1.png',gradX1, cmap='gray')
    #plt.imshow(gradX1, cmap='gray')
    #plt.show()
    gradX2 = gradient(img2, 0)
    # check
    plt.imsave('results/gradx2.png',gradX2, cmap='gray')
    #plt.imshow(gradX2, cmap='gray')
    #plt.show()

    # compute image gradient in y way
    gradY1 = gradient(img1, 1)
    # check
    plt.imsave('results/grady1.png',gradY1, cmap='gray')
    #plt.imshow(gradY1, cmap='gray')
    #plt.show()
    gradY2 = gradient(img2, 1)
    # check
    plt.imsave('results/grady2.png',gradY2, cmap='gray')
    #plt.imshow(gradY2, cmap='gray')
    #plt.show()

    # create the gradient of the desired image
    gradX = imnorm(gradX1 + gradX2)
    gradY = imnorm(gradY1 + gradY2)
    #gradX = np.maximum(gradX1, gradX2)
    #gradY = np.maximum(gradY1, gradY2)

    # check
    plt.imsave('results/gradxcomb.png', gradX, cmap='gray')
    plt.imsave('results/gradycomb.png', gradY, cmap='gray')
    #plt.imshow(gradX, cmap='gray')
    #plt.show()
    #plt.imshow(gradY, cmap='gray')
    #plt.show()

    # use Gauss-Seidel iterations to reconstruct the image from gradient
    result = img1.copy()
    
    Gx2 = gradX.copy()
    padding = np.full((1,sizey), 0)
    Gx2 = Gx2[1:, :]
    Gx2 = np.vstack([Gx2, padding])

    Gy2 = gradY.copy()
    padding = np.full((sizex,1), 0)
    Gy2 = Gy2[:,1:]
    Gy2 = np.hstack([Gy2, padding])

    # divergence divG
    divG = gradX-Gx2 + gradY-Gy2

    # pad around the image
    #result = np.vstack([np.zeros((1,sizey, channels)), result, np.zeros((1,sizey, channels))])
    #result = np.hstack([np.zeros((sizex+2,1, channels)), result, np.zeros((sizex+2,1, channels))])
    #result[:,0,:] = result[:,1,:]
    #result[:,-1,:] = result[:,-2,:]
    #result[0,:,:] = result[1,:,:]
    #result[-1,:,:] = result[-2,:,:]


    it = 0
    while it < 200:
        for i in range(0,sizex-1):
            for j in range(0,sizey-1):
                r = (result[i+1, j] + result[i-1, j] + result[i, j+1] + result[i, j-1])
                r -= divG[i, j]
                r *= 0.25
                result[i, j] = r
        it += 1

    result = imnorm(result)
    plt.imsave('results/' + img_name + '.png', result)
    plt.imshow(result)

    print("Done! See the results/images/ folder for the saved image.")
    # show figure
    plt.show()