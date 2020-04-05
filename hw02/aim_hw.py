import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import os
import itertools

if __name__ ==  "__main__":
    print("Welcome to PPAFFT (Primitive Python Application For Fourier Transform)!")
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

    # fig = plt.figure()

    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis('off')

    f = np.fft.fft2(img)
    fabs = np.abs(f)
    flog = np.log(1 + fabs)
    fimg = flog / np.max(flog)

    plt.subplot(132), plt.imshow(fimg, cmap='gray'), plt.title("Spectrum"), plt.axis('off')

    fshift = np.fft.fftshift(f)
    fabs = np.abs(fshift)
    flog = np.log(1 + fabs)
    fimg = flog / np.max(flog)

    plt.subplot(133), plt.imshow(fimg, cmap='gray'), plt.title("Spectrum with shift"), plt.axis('off')

    # save shifted spectrum image
    plt.imsave('results/'+img_name+'_spectrum.png', fimg, cmap='gray')
    # save full figure with original image and spectrum side by side
    plt.savefig('results/'+img_name+'_figure.png')

    print("Done! See the results/ folder for the saved figure.")

    # show figure
    plt.show()