import glob
import os

from allrgb import AllRGBFilter, FilterError, auto_log

rgb_filter = AllRGBFilter()


if __name__ == '__main__':

    for ref_img in glob.glob(os.path.join('images', '*.jpg')):
        out_img = ref_img[:-4] + '_allrgb.png'
        try:
            rgb_filter.filter_image(ref_img, out_img)
        except FilterError as e:
            auto_log('Error at %s file: %s' % e, level='error')
