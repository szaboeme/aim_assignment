import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import os
import itertools
import math
import networkx as nx
from networkx.algorithms.flow import boykov_kolmogorov


# trying to get more than 2 colors...
def paint ( img, scrib, label1, label2, result ):
    sizex, sizey, channels = img.shape
    # make graph
    G = nx.grid_2d_graph(sizex, sizey)
    G.remove_edges_from(G.edges())
    wmax = 0
    for i in range(sizex):
        for j in range(sizey):
            wm = 0
            # add edges
            if i > 0: # up
                w = 1 + min( sum(img[i-1, j])/3, sum(img[i, j])/3)
                wm += w
                G.add_edge( (i-1, j), (i, j), capacity=w )
                G.add_edge( (i, j), (i-1, j), capacity=w )
            if i < sizex-1: # down
                w = 1 + min( sum(img[i+1, j])/3, sum(img[i, j])/3)
                wm += w
                G.add_edge( (i+1, j), (i, j), capacity=w )
                G.add_edge( (i, j), (i+1, j), capacity=w )
            if j > 0: # left
                w = 1 + min( sum(img[i, j-1])/3, sum(img[i, j])/3)
                wm += w
                G.add_edge( (i, j), (i, j - 1), capacity=w )
                G.add_edge( (i, j - 1), (i, j), capacity=w )
            if j < sizey-1: # right
                w = 1 + min( sum(img[i, j+1])/3, sum(img[i, j])/3)
                wm += w
                G.add_edge( (i, j), (i, j + 1), capacity=w )
                G.add_edge( (i, j + 1), (i, j), capacity=w )
            # find K
            wmax = max(wmax, wm)

    # add s and t nodes
    G.add_node('s')
    G.add_node('t')
    wmax = wmax * 0.05 # lambda

    for i in range(sizex):
        for j in range(sizey):
            # has label1
            if (scrib[i, j] == label1).all():
                G.add_edge( 's', (i, j), capacity=1+wmax )
            else:
                G.add_edge( 's', (i, j), capacity=0 )
            # has label2
            if (scrib[i, j] == label2).all():
                G.add_edge( (i, j), 't', capacity=1+wmax )
            else:
                G.add_edge( (i, j), 't', capacity=0 )

    # solve the min-cut problem
    cutval, partition = nx.minimum_cut(G, 's', 't', flow_func=boykov_kolmogorov)
    s_label, t_label = partition

    for node in s_label:
        if node == 's':
            continue
        i = node[0]
        j = node[1]
        result[i,j,:] = label1
       
    for node in t_label:
        if node == 't':
            continue
        i = node[0]
        j = node[1]
        result[i,j,:] = label2
    return result


if __name__ == "__main__":
    print("Welcome to PPAFIS (Primitive Python Application For Image Segmentation)!")
    imgs = os.listdir('images/')
    #print("\nPlease choose an image from the list to be loaded:\n")

    for i in imgs:
        print(i)
    #img_name = input("\nSelected image name (without '.png'): ")
    img_name = "girl"
    file_name = "images/" + img_name + ".png"
    # check if existing image
    if not os.path.isfile(file_name):
        try:
            sys.exit(0)
        finally:
            print("Invalid file name! Exiting.")

    #img_name2 = input("\nScribble: ")
    img_name2 = "girlscrib"
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
    
    print("\nImage and corresponding scribble loaded.\n")

    # remove alpha channel, if there is one
    if img2.shape[2] > 3:
        scrib = img2[:, :, 0:3]
    else:
        scrib = img2

    # find all the different labels in the scribble
    labels = np.unique( scrib.reshape((scrib.shape[0] * scrib.shape[1], scrib.shape[2])), axis=0 ) 
    labels = [ labels[i] for i in range(len(labels)) ]
    white = labels.pop()

    result = np.ones_like(img1)

    # while there are unused colors, do
    while len(labels) > 1:
        # create a labelling with one color vs. everything else as one label
        newscrib = scrib.copy()
        label1 = labels[0]
        label2 = labels[1]
        for i in range(sizex):
            for j in range(sizey):
                if (scrib[i,j] != label1).any() and (scrib[i,j] != white).any():
                    newscrib[i,j] = label2

        # now solve the mincut with this
        result = paint( img1, newscrib, label1, label2, result )

        # remove painted pixels
        for i in range(sizex):
            for j in range(sizey):
                if (scrib[i,j] != label1).any():
                    scrib[i,j] = white

        # remove the used label
        labels.pop(0)

    # paint the image
    res = img1.copy()
    result = result[:,:,:] * res[:,:,:]

    plt.imsave('results/res_' + img_name + '_' + img_name2 + '.png', result)
    plt.axis('off')
    plt.imshow(result)

    print("Done! See the results/ folder for the saved image.")
    # show figure
    plt.show()

