import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import itertools
import sys

# generate negative image
def negative(im):
    n = im.shape[0]
    res = im.copy()
    for i in range(0, n):
        for j in range(0, n):
            res[i, j] = (1 - im[i, j])
    plt.imsave('result/negative.png', res)

# image thresholding
def threshold(im, t):
    n = im.shape[0]
    m = im.shape[1]
    res = im.copy()
    for i in range(0, n):
        for j in range(0, m):
            if im[i, j, 0] > t or im[i, j, 1] > t or im[i, j, 2] > t:
                res[i, j, :] = 1
                res[i, j, 3] = 1
            else:
                res[i, j, :] = 0
                im[i, j, 3] = 1
    plt.imsave('result/treshold.png', res)

# brightness
def brightness(im, b):
    n = im.shape[0]
    res = im.copy()
    for i in range(0, n):
        for j in range(0, n):
            v = im[i, j, 0] + b
            res[i, j, :] = max(min(v, 1), 0)
    plt.imsave('result/brightness.png', res)

# contrast
def contrast(im, c):
    n = im.shape[0]
    res = im.copy()
    for i in range(0, n):
        for j in range(0, n):
            v = im[i, j, 0] * c
            res[i, j, :] = max(min(v, 1), 0)
    plt.imsave('result/contrast.png', res)

# gamma correction
def gamma(im, g):
    n = im.shape[0]
    res = im.copy()
    for i in range(0, n):
        for j in range(0, n):
            res[i, j] = pow(im[i, j], g)
    plt.imsave('result/gamma.png', res)

# non-linear contrast
def nonlincontrast(im, alpha):
    gamma = 1 / (1 - alpha)
    n = im.shape[0]
    res = im.copy()
    for i in range(0, n):
        for j in range(0, n):
            if im[i, j, 0] < 0.5:
                newvalue = 0.5 * pow(2*im[i, j, 0], gamma)
            else:
                newvalue = 1 - (0.5 * pow(2 - 2*im[i, j, 0], gamma))
            res[i, j] = newvalue
    plt.imsave('result/nonlin_contrast.png', res)

# check user value for operation, exit if wrong
def checkval(value, lo, hi):
    if value < lo or value > hi:
        try:
            sys.exit(0)
        finally:
            print("Invalid value! Exiting.")

if __name__ ==  "__main__":    
    print("Welcome to PPAFMO (Primitive Python Application For Monadic Operations)!")

    img_name = input("Please provide image name to be loaded: ")
    # check if existing image
    if not os.path.isfile(img_name):
        try:
            sys.exit(0)
        finally:
            print("Invalid file name! Exiting.")

    # read image
    #img = plt.imread( 'Lena.png' )
    img = plt.imread( img_name )

    # show original
    # plt.suptitle("Loaded image")
    # plt.imshow( img )
    # plt.show()

    print("Select desired image operation. Type in ")
    print("  neg   for negative image")
    print("  tr    for tresholding")
    print("  br    for brightness change")
    print("  con   for contrast change")
    print("  gam   for gamma correction")
    print("  nlc   for non-linear contrast change")
    op = input("I want to do: ")

    # decide what to do
    # negative
    if op == 'neg':
        negative(img)
    # threshold
    elif op == 'tr':
        val = float(input("Provide threshold value: "))
        checkval(val, 0.0, 1.0)
        threshold(img, val)
    # brightness
    elif op == 'br':
        val = float(input("Provide brightness value: "))
        checkval(val, -1.0, 1.0)
        brightness(img, val)
    # contrast
    elif op == 'con':
        val = float(input("Provide contrast value: "))
        checkval(val, 0.0, 1000.0)
        contrast(img, val)
    # gamma 
    elif op == 'gam':
        val = float(input("Provide gamma value: "))
        checkval(val, 0.0, 1000.0)
        gamma(img, val)
    # non-linear contrast
    elif op == 'nlc':
        val = float(input("Provide alpha value: "))
        checkval(val, 0.0, 1.0)
        nonlincontrast(img, val)
    # non-existent operation
    else:
        try:
            sys.exit(0)
        finally:
            print("Invalid operation! Exiting.")

    # operation ok
    print("Done! Result saved to 'result/operation_name.png'.")