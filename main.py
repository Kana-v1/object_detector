from CNN.layers.conv_layer import ConvLayer
import numpy as np
import cv2

l = ConvLayer(3, 3, 2, 1)

img = cv2.imread(r'/Users/maksymsolomodenko/Desktop/conv_image_test.jpg')

weights, activations = l.forward(img)

print(weights.shape)
print(weights)
