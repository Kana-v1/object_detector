from CNN.layers.conv_layer import ConvLayer
import numpy as np

l = ConvLayer(2, 3, 1, 0)

img = np.ones([3, 3, 1], dtype=np.uint8)

# print(l.weights)
# print(l.weights.shape)
