import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


def prepare(img_path):
    img = load_img(img_path, target_size=(150, 150))   
    x = img_to_array(img)     
    x = x.reshape((1,) + x.shape)  

    return x


new_model = tf.keras.models.load_model('model')  
print (new_model.summary())

x = prepare(sys.argv[-1])



 


print (new_model.predict(x))
