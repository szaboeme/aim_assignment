import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import os
import itertools
import math


# normalize image to 0..1 range
def imnorm(img):
    minr = np.amin(img[:,:,0])
    ming = np.amin(img[:,:,1])
    minb = np.amin(img[:,:,2])
    maxr = np.amax(img[:,:,0])
    maxg = np.amax(img[:,:,1])
    maxb = np.amax(img[:,:,2])
    result = img.copy()
    result[:,:,0] = (img[:,:,0] - minr) * (1.0 / (maxr - minr))
    result[:,:,1] = (img[:,:,1] - ming) * (1.0 / (maxg - ming))
    result[:,:,2] = (img[:,:,2] - minb) * (1.0 / (maxb - minb))
    #result = (img - np.min(img)) * (1.0 / (np.max(img) - np.min(img)))
    return result

# computes normalized gradients from img
def gradients(img):
    sizex, sizey, channels = img.shape
    Gx = np.zeros_like(img)
    Gy = np.zeros_like(img)

    #gx[x,y] = I[x+1,y]-I[x,y]
    #gy[x,y] = I[x,y+1]-I[x,y]

    for i in range(sizex - 1):
        for j in range(sizey):
            Gx[i,j,0] = img[i + 1,j,0] - img[i,j,0]
            Gx[i,j,1] = img[i + 1,j,1] - img[i,j,1]
            Gx[i,j,2] = img[i + 1,j,2] - img[i,j,2]

    for i in range(sizex):
        for j in range(sizey - 1):
            Gy[i, j,0] = img[i,j + 1,0] - img[i,j,0]
            Gy[i, j,1] = img[i,j + 1,1] - img[i,j,1]
            Gy[i, j,2] = img[i,j + 1,2] - img[i,j,2]
    
    return Gx, Gy

if __name__ == "__main__":
    print("Welcome to PPAFIE (Primitive Python Application For Image Editing)!")
    imgs = os.listdir('images/')
    print("\nPlease choose an image from the list to be loaded:\n")
    for i in imgs:
        print(i)
    #img_name = input("\nSelected image name (without '.png'): ")
    img_name = "sunder"
    file_name = "images/" + img_name + ".png"
    # check if existing image
    if not os.path.isfile(file_name):
        try:
            sys.exit(0)
        finally:
            print("Invalid file name! Exiting.")

    #img_name2 = input("\nSelect another image (of the same size): ")
    img_name2 = "sover"
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
    gradX1, gradY1 = gradients(img1)
    gradX2, gradY2 = gradients(img2)
    # save gradients as images
    plt.imsave('results/gradx1_' + img_name + '.png', imnorm(gradX1), cmap='gray')
    plt.imsave('results/gradx2_' + img_name + '.png', imnorm(gradX2), cmap='gray')
    plt.imsave('results/grady1_' + img_name + '.png', imnorm(gradY1), cmap='gray')
    plt.imsave('results/grady2_' + img_name + '.png', imnorm(gradY2), cmap='gray')

    # create the gradient of the desired image
    gradX = gradX1.copy()
    gradY = gradY1.copy()
    for i in range(sizex):
        for j in range(sizey):
            # nr, ng, nb = sqrt(gx.r^2 + gy.r^2), sqrt(gx.g^2 + gy.g^2),
            # sqrt(gx.b^2 + gy.b^2)
            n1r = np.sqrt(gradX1[i,j,0] ** 2 + gradY1[i,j,0] ** 2)
            n1g = np.sqrt(gradX1[i,j,1] ** 2 + gradY1[i,j,1] ** 2)
            n1b = np.sqrt(gradX1[i,j,2] ** 2 + gradY1[i,j,2] ** 2)
            n2r = np.sqrt(gradX2[i,j,0] ** 2 + gradY2[i,j,0] ** 2)
            n2g = np.sqrt(gradX2[i,j,1] ** 2 + gradY2[i,j,1] ** 2)
            n2b = np.sqrt(gradX2[i,j,2] ** 2 + gradY2[i,j,2] ** 2)
            # red
            if n1r + n1g + n1b < n2r + n2g + n2b:
                gradX[i,j,:] = gradX2[i,j,:]
                gradY[i,j,:] = gradY2[i,j,:]
            else:
                gradX[i,j,:] = gradX1[i,j,:]
                gradY[i,j,:] = gradY1[i,j,:]

    # check
    plt.imsave('results/gradxcomb_' + img_name + '.png', imnorm(gradX), cmap='gray')
    plt.imsave('results/gradycomb_' + img_name + '.png', imnorm(gradY), cmap='gray')


    # compute divergence
    #divG = gx[x+1,y]-gx[x,y] + gy[x,y+1]-gy[x,y]
    divG = gradX.copy()
    for i in range(sizex-1):
        for j in range(sizey-1):
            divG[i,j,0] = gradX[i,j,0] - gradX[i-1,j,0] + gradY[i,j,0] - gradY[i,j-1,0]
            divG[i,j,1] = gradX[i,j,1] - gradX[i-1,j,1] + gradY[i,j,1] - gradY[i,j-1,1]
            divG[i,j,2] = gradX[i,j,2] - gradX[i-1,j,2] + gradY[i,j,2] - gradY[i,j-1,2]
            
            #divG[i,j] = gradX[i+1,j] - gradX[i,j] + gradY[i,j+1] - gradY[i,j]
    
    # zero image with borders copied from the original
    #result = img1.copy()
    result = np.zeros_like(img2)
    result[0,:,:] = img2[0,:,:]
    result[:,0,:] = img2[:,0,:]
    result[-1,:,:] = img2[-1,:,:]
    result[:,-1,:] = img2[:,-1,:]
    it = 0

    # use Gauss-Seidel iterations to reconstruct the image from gradient
    while it < 200000:
        it += 1
        #for i in range(1,sizex - 1):
        #    for j in range(1,sizey - 1):
        #        result[i, j, 0] = 0.25 * (result[i + 1, j, 0] + result[i - 1, j, 0] + result[i, j + 1, 0] + result[i, j - 1, 0] - divG[i, j, 0])
        #        result[i, j, 1] = 0.25 * (result[i + 1, j, 1] + result[i - 1, j, 1] + result[i, j + 1, 1] + result[i, j - 1, 1] - divG[i, j, 1])
        #        result[i, j, 2] = 0.25 * (result[i + 1, j, 2] + result[i - 1, j, 2] + result[i, j + 1, 2] + result[i, j - 1, 2] - divG[i, j, 2])

        result[1:-1, 1:-1, :] = 0.25 * (result[0:-2, 1:-1, :] + result[2: , 1:-1, :] + result[1:-1, 0:-2, :] + result[1:-1, 2: , :] - divG[1:-1,1:-1, :])
        
        if it % 10000 == 0:
            plt.imsave('results/res_' + img_name + '_' + str(it) + '.png', imnorm(result))

    result2 = imnorm(result)
    plt.imsave('results/res_' + img_name + '.png', imnorm(result2))
    plt.axis('off')
    plt.imshow(result2)

    print("Done! See the results/ folder for the saved image.")
    # show figure
    plt.show()