import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import os
import itertools
import math

# create a 1D Gaussian kernel
def gauss1d (sigma, s):
    size = int(s/2)
    result = []

    #result = [ (1.0/(sigma * math.sqrt(2.0*np.pi))) * (math.exp(-(i*i)/(2.0*sigma*sigma))) for i in range(-size, size+1) ]
    for i in range(-size, size+1):
        result.append( (1.0/(sigma * math.sqrt(2.0*np.pi))) * (math.exp(-(i*i)/(2.0*sigma*sigma))) )
    ss = sum(result)

    #return np.array([result[i]/ss for i in range(len(result))])
    for k in range(len(result)):
        result[k] /= ss
    
    return result

if __name__ ==  "__main__":
    print("Welcome to PPAFGC (Primitive Python Application For Gaussian Convolution)!")
    imgs = os.listdir('images/')
    print("\nPlease choose an image from the list to be loaded:\n")
    for i in imgs:
        print(i)
    img_name = input("\nSelected image name (without '.png'):  ")
    file_name = "images/" + img_name + ".png"

    # check if existing image
    if not os.path.isfile(file_name):
        try:
            sys.exit(0)
        finally:
            print("Invalid file name! Exiting.")

    # read chosen image
    img = plt.imread( file_name )

    # if more than 2 image dimensions
    if (img.ndim > 2):
        img = img[:,:,0] # take one channel only

    sig = float(input("\nSigma value: "))
    fsize = int(input("\nFilter size: "))

    # create 1D filter
    filter = gauss1d(sig, fsize)
    #print(filter)

    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis('off')

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

    # apply in one direction
    for i in range(imsize):
        for j in range(pad, imsize+pad):
            #part = result[i, j-pad:j+pad+1] 
            #part = padded[i-pad:i+pad+1, j]
            #r = part @ np.transpose(filter)
            r = 0
            for k in range(fsize):
                r += result[i, j-pad+k] * filter[k]
            result[i, j] = r

    result = result[:, pad:imsize+pad]
    plt.imsave('results/'+img_name+'1dir'+'_sig'+str(sig)+'_size'+str(fsize)+'.png', result, cmap='gray')
    plt.subplot(132), plt.imshow(result, cmap='gray'), plt.title("One direction"), plt.axis('off')

    leftpadding = []
    rightpadding = []
    for k in range(pad):
        leftpadding.append(result[0, :])
        rightpadding.append(result[imsize-1, :])

    leftpadding = np.reshape(np.array(leftpadding), (pad, imsize))
    rightpadding = np.reshape(np.array(rightpadding), (pad, imsize))
    #padded = result[pad:imsize+pad, :]
    padded = np.vstack([leftpadding, result, rightpadding])
    result2 = padded.copy()

    # apply in other direction
    for i in range(pad, imsize+pad):
        for j in range(imsize):
            #part = padded[i-pad:i+pad+1, j]
            #part = result[i, j-pad:j+pad+1] 
            #r = filter @ part
            r = 0
            for k in range(fsize):
                r += padded[i-pad+k, j] * filter[k]
            result2[i, j] = r

    result2 = result2[pad:imsize+pad, :]
    plt.subplot(133), plt.imshow(result2, cmap='gray'), plt.title("Both directions"), plt.axis('off')

    # save shifted spectrum image
    plt.imsave('results/'+img_name+'_sig'+str(sig)+'_size'+str(fsize)+'.png', result2, cmap='gray')
    # save full figure with original image and spectrum side by side
    # plt.savefig('results/'+img_name+'_full.png')

    print("Done! See the results/ folder for the saved image.")

    # show figure
    plt.show()