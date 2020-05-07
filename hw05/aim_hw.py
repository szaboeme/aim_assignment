import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import os
import itertools
import math


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
    imcopy = rgb2gray(img.copy())
    result = imcopy.copy()
    imcopy2 = imcopy.copy()
    if ax == 0:
        padding = np.full((1,sizey), 0)
        imcopy = np.vstack([padding, imcopy])
        imcopy2 = np.vstack([imcopy2, padding])
        for i in range(sizex):
            for j in range(sizey):
                result[i, j] = imcopy[i,j] - imcopy2[i,j]
        #imcopy = np.vstack([padding, imcopy])
        #imcopy2 = np.vstack([imcopy2, padding])
        #result = imcopy[:,:] - imcopy2[:,:]
        #result = result[:-1,:]
    elif ax == 1:
        padding = np.full((sizex,1), 0)
        imcopy = np.hstack([padding, imcopy])
        imcopy2 = np.hstack([imcopy2, padding])
        for i in range(sizex):
            for j in range(sizey):
                result[i, j] = imcopy[i,j] - imcopy2[i,j]
        #imcopy = np.hstack([padding, imcopy])
        #imcopy2 = np.hstack([imcopy2, padding])
        #result = imcopy[:,:] - imcopy2[:,:]
        #result = result[:,:-1]

    return result

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
    plt.imsave('results/gradx1.png',gradX1, cmap='gray')
    gradX2 = gradient(img2, 0)
    plt.imsave('results/gradx2.png',gradX2, cmap='gray')
    # compute image gradient in y way
    gradY1 = gradient(img1, 1)
    plt.imsave('results/grady1.png',gradY1, cmap='gray')
    gradY2 = gradient(img2, 1)
    plt.imsave('results/grady2.png',gradY2, cmap='gray')

    # create the gradient of the desired image
    gradX = gradX1
    gradY = gradY1
    for i in range(sizex):
        for j in range(sizey):
            n1 = np.sqrt(gradX1[i,j]**2 + gradY1[i,j]**2)
            n2 = np.sqrt(gradX2[i,j]**2 + gradY2[i,j]**2)
            if n1 < n2:
                gradX[i,j] = gradX2[i,j]
                gradY[i,j] = gradY2[i,j]
            else:
                gradX[i,j] = gradX1[i,j]
                gradY[i,j] = gradY1[i,j]

    # check
    plt.imsave('results/gradxcomb.png', gradX, cmap='gray')
    plt.imsave('results/gradycomb.png', gradY, cmap='gray')

    #divG = gradX[1:-2,1:-2] - gradX[2:-1,1:-2] + gradY[1:-2,1:-2] - gradY[1:-2,2:-1]

    result = img1.copy()
    it = 0
    # use Gauss-Seidel iterations to reconstruct the image from gradient
    while it < 100000:
        it += 1
        for i in range(1,sizex-1):
            for j in range(1,sizey-1):
                r = (result[i+1, j] + result[i-1, j] + result[i, j+1] + result[i, j-1])
                r -= (gradX[i,j] - gradX[i+1,j] + gradY[i,j] - gradY[i,j+1])
                r *= 0.25
                result[i, j] = r

        #result[1:-2, 1:-2, 0] = 0.25*(result[0:-3, 1:-2, 0] + result[1:-2, 0:-3, 0] + result[2:-1, 1:-2, 0] + result[1:-2, 2:-1, 0] - divG[:,:])
        #result[1:-2, 1:-2, 1] = 0.25*(result[0:-3, 1:-2, 1] + result[1:-2, 0:-3, 1] + result[2:-1, 1:-2, 1] + result[1:-2, 2:-1, 1] - divG[:,:])
        #result[1:-2, 1:-2, 2] = 0.25*(result[0:-3, 1:-2, 2] + result[1:-2, 0:-3, 2] + result[2:-1, 1:-2, 2] + result[1:-2, 2:-1, 2] - divG[:,:])
        
        if it % 10000 == 0:
            plt.imsave('results/'+img_name+'_'+str(it)+'.png', imnorm(result.copy()))

    result = imnorm(result)
    plt.imsave('results/' + img_name + str(it) + '.png', result)
    plt.imshow(result)

    print("Done! See the results/ folder for the saved image.")
    # show figure
    plt.show()