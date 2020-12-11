import cv2
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


def gauss2D(shape=(3,3),sigma=0.5):

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def rotate (image_in, R='0'):
    if R== '0':
        return image_in
    if R == '90' :
        img_rotate_90_clockwise = cv2.rotate ( image_in , cv2.ROTATE_90_CLOCKWISE )
        return img_rotate_90_clockwise
    if R == '180' :
        img_rotate_180 = cv2.rotate ( image_in , cv2.ROTATE_180 )
        return img_rotate_180
    if R == '270' :
        img_rotate_90_counterclockwise = cv2.rotate ( image_in , cv2.ROTATE_90_COUNTERCLOCKWISE )
        return img_rotate_90_counterclockwise


def image_minp(image_in, R= '0' , type='lowpass' ):
    if (type == 'lowpass' or type == 'none'):
        m , n, k = image_in.shape

        # Develop Averaging filter(3, 3) mask
        mask = np.ones ( [ 3 , 3 ] , dtype = int )
        mask = mask / 9

        # Convolve the 3X3 mask over the image
        img_new = np.zeros ( [ m , n, k ] )

        for i in range ( 1 , m - 1 ) :
            for j in range ( 1 , n - 1 ) :
                temp = image_in [ i - 1 , j - 1 ] * mask [ 0 , 0 ] + image_in [ i - 1 , j ] * mask [ 0 , 1 ] + image_in [
                    i - 1 , j + 1 ] * \
                       mask [ 0 , 2 ] + image_in [ i , j - 1 ] * mask [ 1 , 0 ] + image_in [ i , j ] * mask [ 1 , 1 ] + image_in [
                           i , j + 1 ] * mask [ 1 , 2 ] + image_in [ i + 1 , j - 1 ] * mask [ 2 , 0 ] + image_in [ i + 1 , j ] * \
                       mask [
                           2 , 1 ] + image_in [ i + 1 , j + 1 ] * mask [ 2 , 2 ]

                img_new [ i , j ] = temp

        img_new = img_new.astype ( np.uint8 )
        img_new= rotate(img_new, R)
        return img_new
    if type == 'median':
        image_in = cv2.imread(imag,0)
        m , n = image_in.shape

        # Traverse the image. For every 3X3 area,
        # find the median of the pixels and
        # replace the ceter pixel by the median
        img_new1 = np.zeros ( [ m , n] )

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = [image_in[i - 1, j - 1],
                        image_in[i - 1, j],
                        image_in[i - 1, j + 1],
                        image_in[i, j - 1],
                        image_in[i, j],
                        image_in[i, j + 1],
                        image_in[i + 1, j - 1],
                        image_in[i + 1, j],
                        image_in[i + 1, j + 1]]

                temp = sorted(temp)
                img_new1[i, j] = temp[4]

        img_new1 = img_new1.astype ( np.uint8 )
        img_new1 = rotate ( img_new1 , R )
        return img_new1

    if type == 'gaussian' :
        m , n, k = image_in.shape

        # Develop Averaging filter(3, 3) mask
        mask = gauss2D((3,3), 1)


        # Convolve the 3X3 mask over the image
        img_new = np.zeros ( [ m , n, k ] )

        for i in range ( 1 , m - 1 ) :
            for j in range ( 1 , n - 1 ) :
                temp = image_in [ i - 1 , j - 1 ] * mask [ 0 , 0 ] + image_in [ i - 1 , j ] * mask [ 0 , 1 ] + \
                       image_in [
                           i - 1 , j + 1 ] * \
                       mask [ 0 , 2 ] + image_in [ i , j - 1 ] * mask [ 1 , 0 ] + image_in [ i , j ] * mask [ 1 , 1 ] + \
                       image_in [
                           i , j + 1 ] * mask [ 1 , 2 ] + image_in [ i + 1 , j - 1 ] * mask [ 2 , 0 ] + image_in [
                           i + 1 , j ] * \
                       mask [
                           2 , 1 ] + image_in [ i + 1 , j + 1 ] * mask [ 2 , 2 ]

                img_new [ i , j ] = temp

        img_new = img_new.astype ( np.uint8 )
        img_new = rotate ( img_new , R )
        return img_new

def predict (x):
    result = new_model.predict(x)

    if (result[0]  > .5):
        print ('dog')
    else: print ('cat')


## loading model
new_model = tf.keras.models.load_model('model')  
#print (new_model.summary())

print ("""
    ____  _____ ____     ____  ____  ____      ____________________   
   / __ \/ ___// __ \   / __ \/ __ \/ __ \    / / ____/ ____/_  __/   
  / / / /\__ \/ /_/ /  / /_/ / /_/ / / / /_  / / __/ / /     / /      
 / /_/ /___/ / ____/  / ____/ _, _/ /_/ / /_/ / /___/ /___  / /       
/_____//____/_/      /_/   /_/ |_|\____/\____/_____/\____/ /_/        
                                                                   
""")

print("""
welcome at our dsp project

this is a tool to classify cats/dogs images
it starts by performing some preprocessing then it inputs this image into a pre-trained CNN

for our model to perform well you are adviced to enter a non-noise image in landscape mode.

you can use our options to do the needed rotation and filtrations

there is limitations to our model due to the size of the training dataset and size of the network 
so please execuse any inaccuacy in classifications

also, plz enter valid image path for our program to produce sensible results  

""" )



imag= input ('The image path: ')
type = input ("Type of the filter (none , lowpass, gaussian): ")
r= input ("Enter the degree of rotation (0, 90, 180, 270) if no rotation needed enter 0 : \n")



img = cv2.imread(imag)

out= image_minp(img, r, type)

out_resized = cv2.resize(out, (150,150), interpolation = cv2.INTER_AREA)
x = img_to_array(out_resized)     
x = x.reshape((1,) + x.shape)  

#print (out.shape)
#print (out_resized.shape)

#cv2.imwrite('out.jpg', out )

predict(x)
