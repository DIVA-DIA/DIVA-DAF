import argparse
import matplotlib
import matplotlib.cm
import numpy as np
import os

from PIL import Image


def visualize(img, out):
    read_img = np.asarray(Image.open(img))
    # class_encodings = [(1, 'Background'), (2, 'Comment'), (4, 'Decoration'), (6, 'Comment + Decoration'),
    #                    (8, 'Main Text'), (10, 'Main Text + Comment'), (12, 'Main Text + Decoration')]
    class_encodings = [(1, 'Background', (0, 0, 0)),
                       (2, 'Comment', (255, 0, 0)),
                       (4, 'Decoration', (255, 255, 0)),
                       (6, 'Comment + Decoration', (125, 125, 125)),
                       (8, 'Main Text', (255, 0, 255)),
                       (10, 'Main Text + Comment', (0, 0, 255)),
                       (12, 'Main Text + Decoration', (0, 255, 255))]

    dest_filename = out  # os.path.join('images', os.path.basename(img)[:-4], 'viz_' + os.path.basename(img))
    if not os.path.exists(os.path.dirname(dest_filename)):
        os.makedirs(os.path.dirname(dest_filename))

    img_la = np.zeros(read_img.shape)

    # Extract just blue channel
    # out_blue = read_img[:, :, 2] #[2140:2145, 1570:1575]
    gt_blue = np.copy(read_img[:, :, 2]) #[2140:2145, 1570:1575]

    # Get boundary pixels and adjust the gt_image for the border pixel -> set to background (1)
    boundary_mask = read_img[:, :, 0].astype(np.uint8) == 128
    gt_blue[boundary_mask] = 1

    # subtract the boundary pixel from the gt
    boundary_pixel = read_img[:, :, 0].astype(np.uint8) == 128
    gt_blue[boundary_pixel] = 1

    # Colours are in RGB
    cmap = matplotlib.cm.get_cmap('Spectral')
    # colors = [cmap(i / len(class_encodings), bytes=True)[:3] for i in range(len(class_encodings))]
    colors = [c[2] for c in class_encodings]

    # Get the mask for each colour
    masks = {color: (gt_blue == i[0]) > 0 for color, i in zip(colors, class_encodings)}

    # colour the pixels according to the masks
    for c, mask in masks.items():
        img_la[mask] = c


    # img = np.copy(read_img)
    # blue = read_img[:, :, 2]  # Extract just blue channel
    #
    # boundary_pixel = gt_image[:, :, 0].astype(np.uint8) == 128
    # gt_blue[boundary_pixel] = 1
    #
    # # Get boundary pixels and adjust the gt_image for the border pixel -> set to background (1)
    # boundary_mask = gt[:, :, 0].astype(np.uint8) == 128
    #
    # # Colours are in RGB
    # cmap = matplotlib.cm.get_cmap('Spectral')
    # colors = [cmap(i / len(class_encodings), bytes=True)[:3] for i in range(len(class_encodings))]
    #
    # # Get the mask for each colour
    # masks = {color: (blue == i) > 0 for color, i in zip(colors, class_encodings)}
    #
    # # Color the image with relative colors
    # for color, mask in masks.items():
    #     img[mask] = color
    #
    # Make and save the class color encoding
    color_encoding = {str(i[1]): color for color, i in zip(colors, class_encodings)}

    make_colour_legend_image(os.path.join(os.path.dirname(dest_filename), "output_visualizations_colour_legend.png"),
                             color_encoding)

    # Write image to output folder
    Image.fromarray(img_la.astype(np.uint8)).save(dest_filename)


def make_colour_legend_image(img_name, colour_encoding):
    import matplotlib.pyplot as plt

    labels = sorted(colour_encoding.keys())
    colors = [tuple(np.array(colour_encoding[k])/255) for k in labels]
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", c) for c in colors]
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(img_name, dpi=1000, bbox_inches=bbox)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img',
                        help='Path to the gt image',
                        type=str,
                        required=True)
    parser.add_argument('-o', '--out',
                        help='Path to output image',
                        type=str,
                        required=True)
    args = parser.parse_args()
    visualize(**args.__dict__)
