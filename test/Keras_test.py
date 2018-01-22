import numpy as np
from keras.preprocessing import image

CURRENT_DIRECTORY = "C:/Users/I347798/git/project_tomato/"
IMAGES_DIRECTORY = "images/"

path = CURRENT_DIRECTORY + IMAGES_DIRECTORY

image_path = "C:/Users/I347798/git/project_tomato/images/goodtomatos_rot/tomato00.jpg"
print(path)

image = image.load_img(image_path, target_size=(100, 100))

print(image)