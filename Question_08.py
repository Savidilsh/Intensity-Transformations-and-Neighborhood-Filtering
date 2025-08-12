import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Zoom function for bilinear and nearest neighbor
def zoom_image(img, scale, method='nearest'):
    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
    if method == 'nearest':
        return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_NEAREST)
    elif method == 'bilinear':
        return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)


# List of image pairs to compare
image_pairs = [
    ('im01.png', 'im01small.png'),
    ('im02.png', 'im02small.png'),
    ('im03.png', 'im03small.png'),
    ('taylor.jpg', 'taylor_small.jpg'),
    ('taylor.jpg', 'taylor_very_small.jpg')  # Assuming the truncated name is 'taylor_very_small.png'
]

# Folder path
folder = r'C:\Users\Savindu Dilshan\Desktop\Github\Intensity-Transformations-and-Neighborhood-Filtering\a1images\a1q5images\\'

# Create a figure with rows for each pair (5 rows x 4 columns)
fig, axs = plt.subplots(len(image_pairs), 4, figsize=(20, 4 * len(image_pairs)))
fig.suptitle('Image Zooming Results (Scale Factor: 4)', fontsize=16)

for idx, (orig_name, small_name) in enumerate(image_pairs):
    # Load small and original images
    img_small = cv.imread(f'{folder}{small_name}', cv.IMREAD_COLOR)
    img_orig = cv.imread(f'{folder}{orig_name}', cv.IMREAD_COLOR)

    # Zoom by factor 4
    zoomed_nn = zoom_image(img_small, 4, 'nearest')
    zoomed_bilinear = zoom_image(img_small, 4, 'bilinear')

    # Plot in the current row
    axs[idx, 0].imshow(cv.cvtColor(img_small, cv.COLOR_BGR2RGB))
    axs[idx, 0].set_title(f'Small Image ({small_name})')
    axs[idx, 0].axis('off')

    axs[idx, 1].imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
    axs[idx, 1].set_title(f'Original Large ({orig_name})')
    axs[idx, 1].axis('off')

    axs[idx, 2].imshow(cv.cvtColor(zoomed_nn, cv.COLOR_BGR2RGB))
    axs[idx, 2].set_title('Zoomed Nearest')
    axs[idx, 2].axis('off')

    axs[idx, 3].imshow(cv.cvtColor(zoomed_bilinear, cv.COLOR_BGR2RGB))
    axs[idx, 3].set_title('Zoomed Bilinear')
    axs[idx, 3].axis('off')

    # Save the current row of plots as a separate image
    # Save the current row of plots as a separate image
    row_fig = plt.figure(figsize=(20, 4))
    for j in range(4):
        plt.subplot(1, 4, j+1)
        if j == 0:
            plt.imshow(cv.cvtColor(img_small, cv.COLOR_BGR2RGB))
            plt.title(f'Small Image ({small_name})')
        elif j == 1:
            plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
            plt.title(f'Original Large ({orig_name})')
        elif j == 2:
            plt.imshow(cv.cvtColor(zoomed_nn, cv.COLOR_BGR2RGB))
            plt.title('Zoomed Nearest')
        else:
            plt.imshow(cv.cvtColor(zoomed_bilinear, cv.COLOR_BGR2RGB))
            plt.title('Zoomed Bilinear')
        plt.axis('off')
    
    plt.tight_layout()
    save_path = r'C:\Users\Savindu Dilshan\Desktop\Github\Intensity-Transformations-and-Neighborhood-Filtering\a1images\comparison_{}.png'
    plt.savefig(save_path.format(small_name.split('.')[0]), bbox_inches='tight', dpi=300)
    plt.close(row_fig)

plt.tight_layout()
plt.show()