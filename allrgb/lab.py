import itertools


def rgb2lab(r, g, b):
    """
    convert RGB to Lab color. Assuming sRGB color space and D65 white point.

    :param r: R component of given color.
    :param g: G component of given color.
    :param b: B component of given color.
    :return: Lab color in a tuple of float numbers.
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # http://www.brucelindbloom.com/index.html?Math.html
    # Inverse sRGB Companding
    r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4

    # http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # sRGB, D65
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
    kappa, epsilon = 903.3, 0.008856

    # http://brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    # White point for D65
    xr, yr, zr = x / 0.95047, y / 1.00000, z / 1.08883

    fx = xr ** (1 / 3.0) if xr > epsilon else (kappa * xr + 16) / 116.0
    fy = yr ** (1 / 3.0) if yr > epsilon else (kappa * yr + 16) / 116.0
    fz = zr ** (1 / 3.0) if zr > epsilon else (kappa * zr + 16) / 116.0

    l = 166.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return l, a, b


def delta_e_76(lab1, lab2):
    """
    Calculate the delta E of two Lab colors using CIE 76 standard.

    :param lab1: color 1
    :param lab2: color 2
    :return: the square of DeltaE (dE^2)
    """

    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    return (l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2


def delta_e_94(lab1, lab2):
    """
    Calculate the delta E of two Lab colors using CIE 94 standard.

    :param lab1: color 1
    :param lab2: color 2
    :return: the square of DeltaE (dE^2)
    """

    # Math from http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    c1 = (a1 ** 2 + b1 ** 2) ** 0.5
    c2 = (a2 ** 2 + b2 ** 2) ** 0.5

    delta_l = l1 - l2
    delta_c = c1 - c2
    delta_h = ((a1 - a2) ** 2 + (b1 - b2) ** 2 - delta_c ** 2)

    kl, sl = 1, 1
    k1, k2 = 0.045, 0.015
    kc, sc = 1, 1 + k1 * c1
    kh, sh = 1, 1 + k2 * c2

    return (delta_l / kl / sl) ** 2 + (delta_c / kc / sc) ** 2 + (delta_h / kh / sh) ** 2


class Rgb2LabMap:
    """ A lookup dictionary for RGB to Lab conversion. """

    def __init__(self, bits=8):
        """
        Create a new RGB to lookup dictionary map.

        :param bits: bit depth of RGB colors
        """

        self.map = {}
        self.bits = bits
        for r, g, b in itertools.product(range(2 ** bits), repeat=3):
            self.map[(r << 2 * bits) + (g << bits) + b] = rgb2lab(r << (8 - bits), g << (8 - bits), b << (8 - bits))

    def get(self, r, g, b):
        """
        Return the Lab equivalence of the given (r, g, b) color. This should give the same result as
        the rgb2lab() function in this file. Just a lot faster.

        :param r: R component of given color.
        :param g: G component of given color.
        :param b: B component of given color.
        :return: Lab color in a tuple of float numbers.
        """

        return self.map[(r << 2 * self.bits) + (g << self.bits) + b]
