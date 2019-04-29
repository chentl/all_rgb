from allrgb import AllRGBFilter

BATCH_NAME = 'NrOv1050d76'

rgb_filter = AllRGBFilter()


def convert(img_name):
    ref_img = 'images/%s.png' % img_name
    out_img = 'images/%s_%s_allrgb.png' % (img_name, BATCH_NAME)

    rgb_filter.filter_image(ref_img, out_img)


if __name__ == '__main__':

    images = ['carlos-quintero', 'chris-holder', 'courtney-hobbs', 'jackson-case',
              'lily-banse', 'marcus-cramer', 'peter-lloyd', 'spencer-davis', 'yolanda-sun']

    for img in images:
        convert(img)
