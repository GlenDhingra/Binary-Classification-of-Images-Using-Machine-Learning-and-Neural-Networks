import numpy as np
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *


def NN_Prediction(path):
    num_px = 64
    parameters = np.load("weights.npy",allow_pickle="TRUE").item()
    fileImage = Image.open(path).convert("RGB").resize([num_px,num_px],Image.ANTIALIAS)
    my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

    image = np.array(fileImage)
    my_image = image.reshape(num_px*num_px*3,1)
    my_image = my_image/255.
    my_predicted_image = predict(my_image, my_label_y, parameters)

    if my_predicted_image[0] == 1:
        print('cat')
        return 'This is an image of a cat'
    else:
        print('dog')
        return 'This is an image of a dog'
    