#Author: Nicholas Rabow
#Description: The purpose of this program is to segment a video that was filmed
# with a high speed camera. A ball was dropped with a noisy background. There are
# 1126 images to process. The goal is to segment the image in order to be able to
# track the center of the ball as well as the height and width.
#Date: 12/8/2021

#Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits import mplot3d
from skimage import filters #gaussian and sobel kernel used from this library
import con_comps

from skimage import morphology

#Name:moving_average
#Parameters: [x]->1D array,[w]->MA size
#Description: Calculates MA of size [w] on signal [x]
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#Name:morpho_dilate
#Parameters:[I]->input image (2D array like),[B]->structing element of MxN size
#           M,N need to be odd. (2D-arraylike)
#Description: Performs morphological dilation on image using SE [B]
#Returns:Image after dilation is perfomed.
#input needs background of zeros and foreground of 1. foreground to be grown
def morpho_dilate(I,B):
    m,n = B.shape #shape of SE
    x,y = I.shape #shape of input image
    p = m//2 #center of SE
    q = n//2 #center of SE
    output = np.zeros((x,y),dtype=bool) #init output array
    I = np.pad(I,((p,p),(q,q)),'constant',constant_values=0)#pad image to allow
    # 'movement'
    for i in range(m): #loop through each element of B
        for j in range(n):
            output |= (I[i:i+x,j:j+y].astype(bool) & B[i,j].astype(bool))
            #Logical and every element in B with shifted input image, this result
            #is or-ed with the current output array
    return output.astype(int) #convert output from bool to int type array

#Name:morpho_erode
#Parameters:[I]->input image (2D array like),[B]->structing element of MxN size
#           M,N need to be odd. (2D-arraylike)
#Description: Performs morphological erosion on image using SE [B]
#Returns:Image after erosion is perfomed.
#input needs background of zeros and foreground of 1. foreground to be shrunken
def morpho_erode(I,B):
    m,n = B.shape #shape of SE
    x,y = I.shape #shape of input image
    p = m//2 #center of SE
    q = n//2 #center of SE
    output = np.ones((x,y),dtype=bool) #init output array
    I = np.pad(I,((p,p),(q,q)),'constant',constant_values=0)#pad image to allow
    # 'movement'
    for i in range(m): #loop through each element of B
        for j in range(n):
            output &= (I[i:i+x,j:j+y].astype(bool) | (not B[i,j]))
            #Logical or every element in (not B) with shifted input image, this
            #result is and-ed with the current output array
    return output.astype(int) #convert output from bool to int type array

#Name:otsu_thresh
#Parameters:[f]->input image (2D-arraylike)
#Description: given input image, calculates otsu threshold
#Returns: otsu threshold (int)
def otsu_thresh(f):
    hist, b_ed = np.histogram(f,bins=256)
    b_c = (b_ed[:-1] + b_ed[1:]) / 2
    hist = hist / np.sum(hist)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    m1 = np.cumsum(hist * b_c) / w1
    m2 = (np.cumsum((hist * b_c)[::-1]) / w2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    class_v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2

    idx = np.argmax(class_v)
    T = b_c[idx]
    return T

###############
#MAIN FUNCTION#
###############
#Psuedocode
#1)Create Replicate Background using difference method
#2)For each image in video:
#   a)Find difference of gaussian(image) and gaussian(background)
#   b)Use sobel kernel for edge Detection
#   c)Apply closing
#   d)Find connected components
#   e)largest component is suspected ball
#   f)find bounds of ball and use them to calculate center, width and height
#3)Plot height,width,and position vs time
def main():
    plot = 0 #used for creating plots of each image
    #CREATE REPLICATE BACKGROUND
    #read first image
    f1 = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S0001000020.tif')
    #read second image
    f2 = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S0001000021.tif')
    x = filters.gaussian(f2,5)-filters.gaussian(f1,5) #take difference
    T = otsu_thresh(x) #thresh the difference image
    x = x > T
    dil = morpho_dilate(x,np.ones((21,21))) #apply closing
    ero = morpho_erode(dil,np.ones((21,21)))
     #use connected components algorithm
    tmp = 0
    a,bc = con_comps.label_cython(ero,background=False,return_num=True)
    for j in range(1,bc+1): #loop to find largest component
        area = np.sum(a[a==j])/j
        if tmp < area:
            tmp = area
            lab_val= j #value of label of largest component
    #find coordinates of all label pixels
    lab_coords = np.array(np.where(a==lab_val))
    rows = lab_coords[0] #x-coordinates
    cols = lab_coords[1] #y-coordinates
    #min row, min col, max row, max col of label
    box1 = [np.min(rows),np.min(cols),np.max(rows),np.max(cols)]
    y = np.mean((box1[0],box1[2])) #y coordinate for center
    x = np.mean((box1[1],box1[3])) #x coordinate for center
    h = box1[2]-box1[0]  #height in pixels
    w = box1[3]-box1[1]  #width in pixels
    #repeat same process as above.
    f3 = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S0001000330.tif')
    f4 = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S0001000220.tif')
    x = filters.gaussian(f4,10)-filters.gaussian(f3,10)
    T = otsu_thresh(x)
    x = x > T
    dil = morpho_dilate(x,np.ones((21,21)))
    ero = morpho_erode(dil,np.ones((21,21)))
    a,bc = con_comps.label_cython(ero,background=False,return_num=True)
    tmp = 0
    for j in range(1,bc+1):
        area = np.sum(a[a==j])/j
        if tmp < area:
            tmp = area
            lab_val= j
    lab_coords = np.array(np.where(a==lab_val))
    rows = lab_coords[0]
    cols = lab_coords[1]
    box2 = [np.min(rows),np.min(cols),np.max(rows),np.max(cols)]
    y = np.mean((box2[0],box2[2]))
    x = np.mean((box2[1],box2[3]))
    h = box2[2]-box2[0]
    w = box2[3]-box2[1]
     #use portion of f1 to replicate background with no foreground
    back = f3
    f3[box2[0]:box2[2],box2[1]:box2[3]] = f1[box2[0]:box2[2],box2[1]:box2[3]]
    plt.figure()
    plt.imshow(back,cmap='gray')
    plt.axis('off')
    plt.title("Noisy Replicate Background")
    plt.show()


    h = np.array([]) #initilize arrays for plotting
    w = np.array([])
    x = np.array([])
    y = np.array([])

    ############
    #IMAGE LOOP#
    ############
    for i in range(1,1127):
    # for n,i in enumerate([2,28,55,72,109,136,163,190,217]): #used for demonstration
        #READ EACH IMAGE
        print(i)
        if i < 10:
            f = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S000100000'+str(i)+'.tif')
        elif i >= 10 and i < 100:
            f = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S00010000'+str(i)+'.tif')
        elif i >= 100 and i < 1000:
            f = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S0001000'+str(i)+'.tif')
        elif i > 1000:
            f = plt.imread('dirtydrop_C001H001S0001\dirtydrop_C001H001S000100'+str(i)+'.tif')


        g = filters.gaussian(f,3)-filters.gaussian(back,3) #Find difference
        g = filters.sobel(g) #apply sobel kernel for edge detection
        T = filters.threshold_otsu(g) #use otsu thresholding
        g = g < T
        dil = morpho_dilate(np.invert(g),np.ones((71,71))) #apply closing
        ero = morpho_erode(dil,np.ones((71,71)))
         #find connected components
        a,bc = con_comps.label_cython(ero,background=False,return_num=True)
        tmp = 0
        for j in range(1,bc+1): #find label with largest area
            area = np.sum(a[a==j])/j
            if tmp < area:
                tmp = area
                lab_val= j #value of pixels in largest label
         #coordinates of all pixels in largest label
        lab_coords = np.array(np.where(a==lab_val))
        rows = lab_coords[0]    #x coordinates of label
        cols = lab_coords[1]    #y coordinates of label
        #find edges of label
        box = [np.min(rows),np.min(cols),np.max(rows),np.max(cols)]
        y = np.append(y,np.mean((box[0],box[2]))) #y coordinate center
        x = np.append(x,np.mean((box[1],box[3]))) #x coordinate center
        h = np.append(h,box[2]-box[0]) #height in pixels
        w  = np.append(w,box[3]-box[1]) #width in pixels

        if plot:
            fig,ax = plt.subplots(1,2)
            plt.suptitle("Frame "+str(i))
            ax[0].imshow(f,cmap='gray')
            ax[0].axis('off')
            y = np.append(y,np.mean((box[0],box[2])))
            x = np.append(x,np.mean((box[1],box[3])))
            h = np.append(h,box[2]-box[0])
            w  = np.append(w,box[3]-box[1])
            rect = (FancyBboxPatch((box[1],box[0]),w[-1],h[-1],
                    fill=False,linewidth=2,edgecolor='red'))
            ax[1].imshow(a)
            ax[1].add_patch(rect)
            ax[1].axis('off')
    #######
    #PLOTS#
    #######
    print("height")
    print(h.shape)
    plt.figure()
    plt.title("Ball Height vs Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Height in Pixels")
    plt.plot(moving_average(h,20),label='20')
    print("width")
    print(w.shape)
    plt.figure()
    plt.title("Ball Width vs Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Width in Pixels")
    plt.plot(moving_average(w,40),label='40')
    print("position")
    print(x.shape,y.shape)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("Position vs Time")
    ax.plot3D(moving_average(x,40),moving_average(y,40),np.arange(1087))
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_zlabel('Frame number')

    plt.show()

if __name__ == '__main__':
    main()
