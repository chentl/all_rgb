# all_rgb

A filter to generate images with all 8-bit RGB colors. Here are some examples:

| Photo by [Chris Holder](https://unsplash.com/photos/pyMNFCC1wqM) | Photo by [Lily Banse](https://unsplash.com/photos/dIRZxtl4mzo) | Photo by [Peter Lloyd](https://unsplash.com/photos/LZ0SEx8NpcU) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![chris-holder_compressed](demo/chris-holder_allrgb_compressed.jpg) | ![chris-holder_compressed](demo/lily-banse_allrgb_compressed.jpg) | ![chris-holder_compressed](demo/peter-lloyd_allrgb_compressed.jpg) |
| [uncompressed version](https://github.com/chentl/all_rgb/blob/master/demo/chris-holder_allrgb.png) | [uncompressed version](https://github.com/chentl/all_rgb/blob/master/demo/lily-banse_allrgb.png) | [uncompressed version](https://github.com/chentl/all_rgb/blob/master/demo/peter-lloyd_allrgb.png) |

## Requirements

- Python 3.4 or higher
- `Numpy` and `Pillow`
- around 24 GBs of free RAM (*I know, I need to optimize the tree structure.* Also this should be runnable on a 16GB RAM machine if it has enough `swap`)
- around 2 GBs of free disk spaces (for cache files).


## Tested environment
- Python 3.7.3 with `numpy (v1.16.2)` and `Pillow (v6.0.0)`


## Usage


### The main script `all_rgb.py`

```bash
$ cd all_rgb
$ python3 all_rgb.py
```

The main script will convert all `.jpg` images inside the `images` folder.

### As a module

```python
# Import All-RGB filter
from allrgb import AllRGBFilter

# Initize a filter instance (Note: for the first run, this may take a while)
rgb_filter = AllRGBFilter()

# Convert a existing image to an All-RGB image.
# The input image must have size of 4096x4096.
rgb_filter.filter_image('images/peter-lloyd.jpg', 'images/peter-lloyd_allrgb.png')
```

You can also convert multiple images in parallel.

```python
from allrgb import AllRGBFilter
from os import path
from multiprocessing import Pool

rgb_filter = AllRGBFilter()
img_list = ['peter-lloyd.jpg', 'lily-banse.jpg', 'chris-holder.jpg']

def convert(img):
    inp_name = path.join('images', img)
    out_name = path.join('images', img[:-4] + '_allrgb.png')
    rgb_filter.filter_image(inp_name, out_name)

with Pool(3) as pool:
    pool.map(convert, img_list)
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

- At the first run, the program needs to initialize and cache the color noises array and the k-D tree. This may take around 15 minutes. For all following runs, caches will be used and initialization will be a lot faster.
- Using a single `AllRGBFilter()` instance for multiple images will be faster than initializing different `AllRGBFilter()` instances for each image.
- After initialization, the filtering process (`filter_image()` method) will takes about 20 minutes for each image (When using a 4GHz CPU).


## Cache files

This program will automatically generate some cache files inside `cache` folder. Those files can be safely deleted after 
applying the filter. 