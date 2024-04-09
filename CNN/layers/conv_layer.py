import numpy as np
from CNN.activations import activations


class ConvLayer:
    def __init__(self, kernel_size: int, kernels_num: int, stride: int):
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size has to be odd number')

        if kernels_num < 1:
            raise ValueError('kernels_num must not be less than 1')

        self.channels_num = 3
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int((kernel_size - 1) / 2)

        std = np.sqrt(2 / kernel_size ** 2 / self.channels_num)
        self.biases = np.random.normal(0, std, size=kernels_num)
        self.kernels = np.array([
            np.random.normal(0, scale=std, size=[kernel_size, kernel_size, self.channels_num])
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

    def backward(self, next_layer_loss: np.ndarray, cur_layer_weights_sum: np.ndarray) -> np.ndarray:
        """calculates back-propagate loss for current layer"""

        loss = np.zeros_like(next_layer_loss)

        for i, kernel in self.kernels:
            flipped_kernel = self.kernels[i, ::-1, ::-1, ...]
            next_loss_i = next_layer_loss[:, :, i]
            for j in range(self.channels_num):
                loss_i = self._full_convolve(next_loss_i, flipped_kernel[:, :, j])
                loss[:, :, j] += loss_i

        loss *= self.activation_fn_grad(cur_layer_weights_sum)

        return loss

    def _pad_img(self, img: np.ndarray, pad_1: int | None = None, pad_2: int | None = None) -> np.ndarray:
        """
        :param pad_1: number of els to pad left/top
        :param pad_2: number of els to pad right/bottom
        """
        if self.padding == 0:
            return img

        if pad_1 is None:
            pad_1 = self.padding

        if pad_2 is None:
            pad_2 = self.padding

        padded_img = np.zeros(tuple(self.padding * 2 + np.array(img.shape[:2])) + img.shape[2:], dtype=img.dtype)
        padded_img[pad_1:-pad_2, pad_1:-pad_2] = img

        return padded_img

    def _stride_img(self, img: np.ndarray, kernel_shape: tuple, stride: int | None = None) -> np.ndarray:
        """""takes an input image and divides it into overlapping patches of a specified window size"""
        x_img_stride, y_img_stride = img.shape[:2]
        x_img_shape, y_img_shape = img.shape[:2]
        x_kernel_shape, y_kernel_shape = kernel_shape[:2]

        if stride is None:
            stride = self.stride

        view_shape = (1 + (x_img_shape - x_kernel_shape) // stride,
                      1 + (y_img_shape - y_kernel_shape) // stride,
                      x_kernel_shape,
                      y_kernel_shape) + img.shape[2:]
        strides = (stride * x_img_stride, stride * y_img_stride, x_img_shape, y_img_shape) + img.strides[2:]

        return np.lib.stride_tricks.as_strided(img, shape=view_shape, strides=strides, writeable=False)

    def _convolve(self, img: np.ndarray, kernel: np.ndarray, stride: int | None = None) -> np.ndarray:
        view = self._stride_img(img, kernel.shape, stride)
        res = np.sum(view * kernel, axis=(2, 3, 4))
        return res

    def _full_convolve(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        img2 = self._inter_leave(img)
        _, pad_left, pad_right = self._compute_full_convolve_shape(img)
        if pad_left or pad_right:
            img2 = self._pad_img(img2, pad_left, pad_right)

        return self._convolve(img2, kernel, stride=1)

    def _inter_leave(self, img: np.ndarray) -> np.ndarray:
        """Interleave array with rows/columns of 0s.
    E.g. stride == 1
        arr = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]
        interLeave(img) ->
        [[1, 0, 2, 0, 3],
         [0, 0, 0, 0, 0],
         [4, 0, 5, 0, 6],
         [0, 0, 0, 0, 0],
         [7, 0, 8, 0, 9]]
    """

        y, x = img.shape[:2]
        to_insert = self.stride - 1
        shape = (y + to_insert * (y - 1), x + to_insert * (x - 1)) + img.shape[:2]

        res = np.zeros(shape)
        res[0::(y + 1), 0::(x + 1), ...] = img
        return res

    def _compute_full_convolve_shape(self, img: np.ndarray) -> tuple[int, int, int]:
        """"
        Compute the shape of a full convolution result

        Returns:
        len_out: length of output img.
        pad_left: number padded to the left in a full convolution.
        pad_right: number padded to the right in a full convolution.

        E.g. img = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        kernel_size = 3
        stride = 2
        A full convolution is done on [*, *, 0], [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8],
        [9, 10, *]. Where * is missing outside the input domain.
        Therefore, the full convolution img has length 6. pad_left = 2, pad_right = 1
        """

        kernel_len = self.kernel_size
        stride = self.stride
        input_img_len = img.shape[0]

        len_out = 1

        idx = 0  # idx of the right end of the kernel

        while True:
            idx += stride
            win_left = idx - kernel_len + 1
            if win_left > input_img_len:
                break

            len_out += 1

        pad_left = kernel_len - 1
        pad_right = idx - kernel_len + 1

        return len_out, pad_left, pad_right
