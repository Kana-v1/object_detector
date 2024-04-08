import numpy as np
from CNN.activations import activations


class ConvLayer:
    def __init__(self, channels_num: int, kernel_size: int, kernels_num: int, stride: int):
        if channels_num != 2 and channels_num != 3:
            raise ValueError('channels must be either 2 or 3')

        if kernel_size % 2 == 0:
            raise ValueError('kernel_size has to be odd number')

        if kernels_num < 1:
            raise ValueError('kernels_num must not be less than 1')

        self.channels_num = channels_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int((kernel_size - 1) / 2)  # to make output the same size as input

        std = np.sqrt(2 / kernel_size ** 2 / channels_num)
        self.biases = np.random.normal(0, std, size=kernels_num)
        self.kernels = np.array([
            np.random.normal(0, scale=std, size=[kernel_size, kernel_size, channels_num])
            for _ in range(kernels_num)
        ])

        self.activation_fn = activations.ReLu
        self.activation_fn_grad = activations.dReLu

    def forward(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(img.shape) != len(self.kernels[0].shape):
            raise ValueError('image shape and filter shape are not equal')

        img_x = img.shape[0]
        img_y = img.shape[1]

        img = self._pad_img(img)

        # weights_sum size
        wx = (img_x + 2 * self.padding - self.kernel_size) // self.stride + 1
        wy = (img_y + 2 * self.padding - self.kernel_size) // self.stride + 1
        wc = len(self.kernels)
        weights_sum = np.zeros([wx, wy, wc])

        for i, kernel in enumerate(self.kernels):
            convolve_sum_up = self._convolve(img, kernel)
            weights_sum[:, :, i] = convolve_sum_up[:, :]

        weights_sum += self.biases
        activation_res = self.activation_fn(weights_sum)

        return weights_sum, activation_res

    def _pad_img(self, img: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return img

        padded_img = np.zeros(tuple(self.padding * 2 + np.array(img.shape[:2])) + img.shape[2:], dtype=img.dtype)
        padded_img[self.padding:-self.padding, self.padding:-self.padding] = img

        return padded_img

    def _stride_img(self, img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
        """""takes an input image and divides it into overlapping patches of a specified window size"""
        x_img_stride, y_img_stride = img.shape[:2]
        x_img_shape, y_img_shape = img.shape[:2]
        x_kernel_shape, y_kernel_shape = kernel_shape[:2]

        view_shape = (1 + (x_img_shape - x_kernel_shape) // self.stride,
                      1 + (y_img_shape - y_kernel_shape) // self.stride,
                      x_kernel_shape,
                      y_kernel_shape) + img.shape[2:]
        strides = (self.stride * x_img_stride, self.stride * y_img_stride, x_img_shape, y_img_shape) + img.strides[2:]

        return np.lib.stride_tricks.as_strided(img, shape=view_shape, strides=strides, writeable=False)

    def _convolve(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        view = self._stride_img(img, kernel.shape)
        res = np.sum(view * kernel, axis=(2, 3, 4))
        return res
