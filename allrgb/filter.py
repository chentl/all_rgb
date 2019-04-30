import glob
import os
import pickle
import random

import numpy as np
from PIL import Image, ImageCms

from .cache import CacheStorage, CacheLoadError
from .kdtree import RGBKdTree
from .log import auto_log

_cache_storage = CacheStorage('filter')

_pattern_dir = os.path.join(os.path.dirname(__file__), 'patterns')
_default_noise_pattern = '256_std20'
_default_index_pattern = '512_std10_mono'


class FilterError(Exception):
    pass


def _sanitize(img):
    """
    Convert image color mode to *RGB*. So it is easier for future
    processing and the returned image is always safe to be saved as PNG file.

    If the input file is in *Lab* color space, it is converted to *sRGB*.

    :param img: Input image as PIL.Image object.
    :return: sanitized image as PIL.Image object.
    """

    if img.mode in ['RGB']:
        img2 = img.copy()
    elif img.mode == 'LAB':
        # PIL cannot convert Lab to RGB automatically.
        # Codes from https://stackoverflow.com/questions/52767317/
        rgb_p = ImageCms.createProfile('sRGB')
        lab_p = ImageCms.createProfile('LAB')
        lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, rgb_p, 'LAB', 'RGB')
        img2 = ImageCms.applyTransform(img, lab2rgb)
    else:
        img2 = img.convert('RGB')

    auto_log('%s(%s) --> %s.' % (img.format, img.mode, img2.mode))

    return img2


def _calc_avg_luminance(sanitized_img):
    """
    Calculate average luminance of a given sanitized image object

    :param sanitized_img: Input image (8-bit file, color mode must be *RGB*).
    :return: average luminance of given image as a float number between 0 and 1.
    """

    from colorsys import rgb_to_hsv

    width, height = sanitized_img.size
    sanitized_data = np.asarray(sanitized_img, dtype=np.double)

    flattened_data = sanitized_data.reshape((width * height, len(sanitized_data[0][0])))

    # Because sanitized_img is always an 8-bit image, we can hard code 255 as limit.
    luminance_data = [rgb_to_hsv(*(x/255.0)[:3])[2] for x in flattened_data]

    # If image has alpha information, alpha value is used to weight average calculation.
    # So transparent pixels will have less influences in calculating average luminance.
    return np.average(luminance_data)


def set_luminance(sanitized_img, target_luminance, mixing_coeff=1.0):
    """
    Set luminance of input image to a target value. Save the result as a new PNG file.

    There are two ways to adjust the luminance of a given image. One is to multiply
    luminance of each pixels with a fixed scaling factor. This will create more nature
    looks, but are easier to be limited by highlight pixels. Another one is to add a
    fixed shift value to luminance of each pixels. This may result in absences of pure
    black and white in the output image.

    This function can use both of those methods, and default to use the "fixed scaling"
    method. you can use the optional argument *mixing_coeff* to control which method are
    used, or use both methods and mix their results together.

    The final luminance *L'* of a pixel which has original luminance *L* is:
    *L' = a \* (L \* scale) + b \* (L + shift)*, where *a* and *b* are weights controlled by
    *mixing_coeff* and *a* + *b* is always 1.

    Note that the average luminance of output image may differ from *target_luminance*.
    When this happens it means there are pixels in the output image that already hit
    pure black or white, so that the luminance adjustment is limited because those pixels
    can not be darker or brighter.

    :param sanitized_img: Input image (8-bit file, color mode must be *RGB*).
    :param target_luminance: target luminance, on scale from 0.0 to 1.0.
    :param mixing_coeff: (optional) control mixing of results of two methods, on scale from -3.0 to 3.0. Default is 1.0,
            which is only use "fixed scaling" method. Value of 0.0 will use the average of
            two methods. Positive value increase weight of "fixed scaling" method, while
            negative value increase weight of "fixed shift" method. Value of 1 and -1 will
            effectively "turn off" the other method. And value beyond 1 or -1 will resulting
            in negative value of *b* or *a*, which usually leads to wired result but can be
            useful in some cases.
    :return: output image as PIL.Image object and the average luminance of output image.
    """

    from colorsys import rgb_to_hsv, hsv_to_rgb

    # Limit target_luminance and mixing_coeff in range
    target_luminance = max(min(1.0, target_luminance), 0.0)
    mixing_coeff = max(min(3.0, mixing_coeff), -3.0)

    sanitized_data = np.asarray(sanitized_img, dtype=np.double)
    modified_img = sanitized_img.copy()
    width, height = sanitized_img.size

    # Calculate average luminance of input image, determine coefficients for luminance transform function.
    input_avg_luminance = _calc_avg_luminance(sanitized_img)
    auto_log('input average luminance: %.4f' % input_avg_luminance)
    auto_log('target average luminance: %.4f' % target_luminance)

    # luminance transform function L' = f(L) = alpha * (L * scale)  +  beta * (L + shift)
    alpha, beta = (1 + mixing_coeff) / 2.0, (1 - mixing_coeff) / 2.0
    scale = target_luminance / input_avg_luminance
    shift = target_luminance - input_avg_luminance
    f = lambda L: alpha * (L * scale) + beta * (L + shift)
    auto_log("mixing_coeff = %.2f, L' = f(L) = %.2f * (L * %.4f) + %.2f * (L + %.4f)" %
             (mixing_coeff, alpha, scale, beta, shift))

    # Function for calculating result data of one pixel
    # Because sanitized_img is always an 8-bit image, we can hard code 255 as limit.
    # 'r', 'g', 'b' for 3 channels in RGB; 'h', 's', 'v' for 3 channels in HSV, 'a' for Alpha channel
    def calc_px(x, y):
        h, s, v = rgb_to_hsv(*(sanitized_data[y][x][:3] / 255.0))
        v = f(v)
        r, g, b = map(int, np.array(hsv_to_rgb(h, s, v)) * 255.0)
        return r, g, b

    # Apply manipulating function to the entire image
    modified_data = [calc_px(x, y) for y in range(height) for x in range(width)]
    modified_img.putdata(modified_data)

    output_avg_luminance = _calc_avg_luminance(modified_img)
    auto_log('output average luminance: %.4f' % output_avg_luminance)
    if abs(target_luminance - output_avg_luminance) > 0.01:
        auto_log('luminance adjustment is limited by highlight and/or shadow clipping.', level='warning')

    return modified_img, output_avg_luminance


class AllRGBFilter:
    """ A filter which can convert any image to all-RGB image. """

    def __init__(self, bits=8, load_patterns=True, noise_pattern=_default_noise_pattern,
                 index_pattern=_default_index_pattern):
        """
        Create a new AllRGBFilter() instance.

        :param bits: bit depth of RGB color. Currently the only allowed value is 8.
        :param load_patterns: Whether auto load pattern images or not.
        :param noise_pattern: Define which pattern is used for color noises. This must be a RGB pattern.
        :param index_pattern: Define which pattern is used for index generations. This must be a Mono-channel pattern.
        """

        assert bits in [8]
        self.bits = bits
        self.size = int(((2 ** bits) ** 3) ** 0.5)
        self._indexes = None
        self._noise_arr_float = None
        self._kd_tree_pickled = None

        auto_log('AllRGBFilter Init: bits = %d, size = %d' % (self.bits, self.size))

        all_patterns = glob.glob(os.path.join(_pattern_dir, 'mat_*.png'))
        self.available_patterns = {os.path.basename(s)[4:-4]: s for s in all_patterns}

        if load_patterns:
            self.load_index_pattern(index_pattern)
            self.load_noise_pattern(noise_pattern)

        self._init_kd_tree()

    @staticmethod
    def _get_pattern_info(pattern_name):
        """ extract metadata from pattern name """

        info = pattern_name.split('_')
        dim, std_int = int(info[0]), int(info[1][3:])
        return dim, std_int

    def load_index_pattern(self, index_pattern):
        """
        Load a pattern image and use it to generations random indexes.

        :param index_pattern: name of the pattern.
        :return: None
        """
        auto_log('load ' + index_pattern)

        assert index_pattern.endswith('_mono')
        assert index_pattern in self.available_patterns

        cache_tag = ('AllRGBFilter', str(self.bits), 'Indexes', index_pattern)
        try:
            self._indexes = _cache_storage.load_cache(cache_tag)
        except CacheLoadError:
            indexes = list(range(self.size * self.size))
            idx_img = Image.open(self.available_patterns[index_pattern])
            idx_arr = list(idx_img.getdata())
            idx_dim, _ = self._get_pattern_info(index_pattern)
            self._indexes = sorted(indexes, key=lambda i: (idx_arr[(i % self.size) % idx_dim +
                                                                   ((i // self.size) % idx_dim) *
                                                                   idx_dim], random.random()))
            _cache_storage.write_cache(cache_tag, self._indexes)

            del indexes, idx_img, idx_arr

    def load_noise_pattern(self, noise_pattern):
        """
        Load a pattern image and use it to generate color noise layer.

        :param noise_pattern: name of the pattern.
        :return: None
        """

        auto_log('load ' + noise_pattern)

        assert not noise_pattern.endswith('_mono')
        assert noise_pattern in self.available_patterns

        cache_tag = ('AllRGBFilter', str(self.bits), 'Noise', noise_pattern)
        try:
            self._noise_arr_float = _cache_storage.load_cache(cache_tag)
        except CacheLoadError:
            noise_pat = Image.open(self.available_patterns[noise_pattern])
            noise_dim, _ = self._get_pattern_info(noise_pattern)

            noise_arr = list(noise_pat.getdata()) * ((self.size // noise_dim) ** 2)
            noise_arr_tiled = [noise_arr[(i % self.size) % noise_dim + ((i // self.size) % noise_dim) * noise_dim]
                               for i in range(self.size * self.size)]
            self._noise_arr_float = np.array(noise_arr_tiled, dtype=np.float)
            _cache_storage.write_cache(cache_tag, self._noise_arr_float)

            del noise_arr, noise_arr_tiled, noise_pat

    def _init_kd_tree(self):
        """ Init. a k-D tree and cache it in memory. """

        auto_log('Init. k-D tree...')
        kd_tree = RGBKdTree(bits=self.bits)
        self._kd_tree_pickled = pickle.dumps(kd_tree, protocol=4)
        del kd_tree

    def filter_image(self, img_name, out_name, target_luminance=None, noise_blend_alpha=0.05, noise_overlay_alpha=0.3):
        """
        Convert a image to an all-RGB image.

        :param img_name: path to input image.
        :param out_name: path to output image.
        :param target_luminance: If being set, will change the brightness of the input image before all-RGB processing,
               this should be used when the input image is too dark to too bright.
        :param noise_blend_alpha: alpha value of blending amount of the color noise layer.
        :param noise_overlay_alpha:  alpha value of overlay blending amount og the color noise layer.
        :return: None
        """

        auto_log('file: %s, target_luminance = %s, noise_blend_alpha = %s, noise_overlay_alpha = %s.' % (img_name,
                 str(target_luminance), str(noise_blend_alpha), str(noise_overlay_alpha)))

        try:
            inp_img = Image.open(img_name)
        except IOError:
            raise FilterError('The file cannot be found, or the image cannot be opened.')
        if inp_img.size != (self.size, self.size):
            raise FilterError('The input image size must be %dx%d' % (self.size, self.size))
        if target_luminance is not None:
            if target_luminance >= 1 or target_luminance <= 0:
                raise FilterError('The target_luminance must be in (0, 1)')
        if noise_blend_alpha > 1 or noise_blend_alpha < 0:
            raise FilterError('The noise_blend_alpha must be in [0, 1]')
        if noise_overlay_alpha > 1 or noise_overlay_alpha < 0:
            raise FilterError('The noise_overlay_alpha must be in [0, 1]')

        sanitized_img = _sanitize(inp_img)

        if target_luminance is not None:
            sanitized_img, _ = set_luminance(sanitized_img, target_luminance)

        inp_arr_float = np.array(list(sanitized_img.getdata()), dtype=np.float)

        if noise_blend_alpha > 0:
            inp_arr_float = inp_arr_float * (1 - noise_blend_alpha) + self._noise_arr_float * noise_blend_alpha
        if noise_overlay_alpha > 0:
            # https://stackoverflow.com/questions/52141987/overlay-blending-mode-in-python-efficiently-as-possible-numpy-opencv
            bg, fg = inp_arr_float / 255.0, self._noise_arr_float / 255.0
            mask = fg >= 0.5
            result = np.zeros_like(bg, dtype=np.float)

            result[~mask] = (2 * bg * fg)[~mask]
            result[mask] = (1 - 2 * (1 - bg) * (1 - fg))[mask]
            inp_arr_float = result * 255.0 * noise_overlay_alpha + inp_arr_float * (1 - noise_overlay_alpha)

        ref_arr = [tuple(p) for p in np.array(inp_arr_float, dtype=np.uint8)]

        ref_name = out_name[:-4] + '_ref.png'
        ref_img = Image.new('RGB', (self.size, self.size))
        ref_img.putdata(ref_arr)
        ref_img.save(ref_name)

        kd_tree = pickle.loads(self._kd_tree_pickled)
        out_arr = [None] * (self.size * self.size)

        for i, index in enumerate(self._indexes):
            if i % 1048576 == 0:
                pct = 100.0 * i / (self.size * self.size)
                auto_log(('%s progress: %.1f%%' % (out_name, pct)))
            out_arr[index] = kd_tree.pop_nearest_neighbor(*ref_arr[index])

        out_img = Image.new('RGB', (self.size, self.size))
        out_img.putdata(out_arr)
        out_img.save(out_name)

        del kd_tree













