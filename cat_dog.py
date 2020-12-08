import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


new_model = tf.keras.models.load_model('model')  

print (new_model.summary())


img = load_img(sys.argv[-1], target_size=(150, 150))   
x = img_to_array(img)     
x = x.reshape((1,) + x.shape)   


print (new_model.predict(x))
