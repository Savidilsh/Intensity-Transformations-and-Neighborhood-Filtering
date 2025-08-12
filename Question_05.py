import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# histogram equalization function
def histogram_equalization(image):
    pixel_counts, intensity_levels = np.histogram(image.flatten(), 256, [0, 256])
    cumulative_sum = pixel_counts.cumsum()
    scaled_sum = cumulative_sum * 255 / cumulative_sum[-1]
    transformed_image = np.interp(image.flatten(), intensity_levels[:-1], scaled_sum)
    return transformed_image.reshape(image.shape).astype(np.uint8)

# Load the image 
img_orig = cv.imread('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/shells.tif', cv.IMREAD_GRAYSCALE)

# Apply equalization
equalized_img = histogram_equalization(img_orig)

# Compute histograms
hist_orig = cv.calcHist([img_orig], [0], None, [256], [0, 256])
hist_eq = cv.calcHist([equalized_img], [0], None, [256], [0, 256])

# Display using subplots
plt.figure(figsize=(20, 4))

plt.subplot(141)
plt.imshow(img_orig, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(142)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(143)
plt.plot(hist_orig)
plt.title('Original Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.subplot(144)
plt.plot(hist_eq)
plt.title('Equalized Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

#save plots
plt.tight_layout()
plt.savefig('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/flatten_histogram.png')
plt.show()