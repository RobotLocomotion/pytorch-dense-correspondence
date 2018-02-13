import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_pixel_correspondence(log_dir, img_a, img_b, uv_a, uv_b, use_previous_plot=None, circ_color='g', show=True):
    if use_previous_plot is None:
        fig, axes = plt.subplots(nrows=2, ncols=2)
    else:
        fig, axes = use_previous_plot[0], use_previous_plot[1]

    img1_filename = log_dir+"images/"+img_a+"_rgb.png"
    img2_filename = log_dir+"images/"+img_b+"_rgb.png"
    img1_depth_filename = log_dir+"images/"+img_a+"_depth.png"
    img2_depth_filename = log_dir+"images/"+img_b+"_depth.png"
    images = [img1_filename, img2_filename, img1_depth_filename, img2_depth_filename]
    images = [mpimg.imread(x) for x in images]
    fig.set_figheight(10)
    fig.set_figwidth(15)
    pixel_locs = [uv_a, uv_b, uv_a, uv_b]
    axes = axes.flat[0:]
    if use_previous_plot is not None:
        axes = [axes[1], axes[3]]
        images = [images[1], images[3]]
        pixel_locs = [pixel_locs[1], pixel_locs[3]]
    for ax, img, pixel_loc in zip(axes[0:], images, pixel_locs):
        ax.set_aspect('equal')
        if isinstance(pixel_loc[0], int) or isinstance(pixel_loc[0], float):
            circ = Circle(pixel_loc, radius=10, facecolor=circ_color, edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
            ax.add_patch(circ)
        else:
            for x,y in zip(pixel_loc[0],pixel_loc[1]):
                circ = Circle((x,y), radius=10, facecolor=circ_color, edgecolor='white', fill=True ,linewidth = 2.0, linestyle='solid')
                ax.add_patch(circ)
        ax.imshow(img)
    if show:
        plt.show()
        return None
    else:
        return fig, axes