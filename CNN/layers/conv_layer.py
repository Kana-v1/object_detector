import numpy as np


class ConvLayer:
    def __init__(self, channels_num: int, kernel_size: int, filters_num: int, stride: int):
        if channels_num != 2 and channels_num != 3:
            raise ValueError('channels must be either 2 or 3')

        if kernel_size % 2 == 0:
            raise ValueError('kernel size has to be odd number')

        if filters_num < 1:
            raise ValueError('filters_num must not be less than 1')

        self.channels_num = channels_num
        self.kernel_size = kernel_size

        weights = 0.01 * np.random.randn(filters_num, channels_num, kernel_size, kernel_size)
        self.weights = weights / np.sqrt(np.prod(weights.shape))
        self.bias = np.zeros(filters_num)

        self.filters = [np.ones((filters_num, channels_num, kernel_size, kernel_size)) for i in range(filters_num)]
        self.stride = stride
        self.padding = int((kernel_size - 1) / 2)  # to make output the same size as input
        self.padding = 1

    def forward(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) != len(self.filters[0].shape):
            raise ValueError('image shape and filter shape are not equal')

        return None

    def _pad_img(self, img: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return img

        padded_img = np.zeros(tuple(self.padding * 2 + np.array(img.shape[:2])) + img.shape[2:], dtype=img.dtype)
        padded_img[self.padding:-self.padding, self.padding:-self.padding] = img

        return padded_img

    def _stride_img(self, img: np.ndarray) -> np.ndarray:
        """"" returns 'view' - parts of the image that equal to the filter size """

    def _convolve(self, img: np.ndarray) -> np.ndarray:
        img_x = img.shape[0]
        img_y = img.shape[1]
        img_c = img.shape[2]

        # output size
        ox = (img_x + 2 * self.padding - self.kernel_size) // self.stride + 1
        oy = (img_y + 2 * self.padding - self.kernel_size) // self.stride + 1
        oc = img_c
        output = np.zeros_like([ox, oy, oc])

        view = self._stride_img(img)
        for i in range(0, len(self.filters) - 1):
            output[i] = np.sum(view * self.filters[i], axis=(2, 3))

        return output
