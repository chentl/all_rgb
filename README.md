# all_rgb

program to generate images with all 8-bit RGB colors.


## Requirements

- Python 3.4 or higher
- `Numpy` and `Pillow`
- around 24 GBs of free RAM (may be runnable on a 16GB `RAM` machine if it has enough `swap`)


## Tested environment
- Python 3.7.3 with `numpy (v1.16.2)` and `Pillow (v6.0.0)`


## Usage

As in `all_rgb.py` script:

```python
# Import All-RGB filter
from allrgb import AllRGBFilter

# Initize a filter instance (Note: for the first run, this may take a few minutes)
rgb_filter = AllRGBFilter()

# Convert a existing image to an All-RGB image.
rgb_filter.filter_image('images/peter-lloyd.png', 'images/peter-lloyd_allrgb.png')
```


## Method

This filter use the following procedures to convert an image to an all-rgb image.

1. Postprocessing image (normalize luminance and add color noises) to `inp_image`;
2. Create a k-D tree holding all 8-bit `RGB` colors and their `La*b*` equivalences;
3. Pick a "random" location `(x, y)`
    - Find the color `C_inp` of `inp_image` at `(x, y)`
    - Pick the nearest color `C_new` from all available colors in the k-D tree for the output image.
4. Repeat step 3 until all locations have been picked.

For `Step 2`, because normal random generator will results in lots of low-frequency components, use [blue noise](https://github.com/MomentsInGraphics/BlueNoise) instead for a more visually pleasing result.

For `Step 3`, compare colors in `Lab` color space instead of `RGB` for a more nature looks.


## Performance

- At the first run, the program needs to initialize and cache the color noises array and the k-D tree. This may takes a few minutes. For all following runs, caches will be used and initialization will be a lot faster.
- Using a single `AllRGBFilter()` instance for multiple images will be faster than initializing different `AllRGBFilter()` instances for each image.
- After initialization, the filtering process (`filter_image()` method) will takes about 30 minutes for each image.

