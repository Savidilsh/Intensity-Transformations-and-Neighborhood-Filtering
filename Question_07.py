import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load the image 
img_orig = cv.imread('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/einstein.png', cv.IMREAD_GRAYSCALE)

# (a) Using filter2D for Sobel
kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel_x_filter2d = cv.filter2D(img_orig, -1, kernel_x)
sobel_y_filter2d = cv.filter2D(img_orig, -1, kernel_y)

# (b) Custom Sobel implementation
def custom_filter2d(img, kernel):
    padded_img = np.pad(img, ((1, 1), (1, 1)), mode='constant')
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = np.sum(padded_img[i:i+3, j:j+3] * kernel)
    return result

sobel_x_custom = custom_filter2d(img_orig, kernel_x)
sobel_y_custom = custom_filter2d(img_orig, kernel_y)

# (c) using the property
row_kernel = np.array([[1], [2], [1]])
col_kernel = np.array([1, 0, -1])
sobel_x_sep = cv.sepFilter2D(img_orig, -1, col_kernel, row_kernel)

# Display using subplots
plt.figure(figsize=(20, 4))

plt.subplot(161)
plt.imshow(img_orig, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(162)
plt.imshow(sobel_x_filter2d, cmap='gray')
plt.title('Sobel X (filter2D)')
plt.axis('off')

plt.subplot(163)
plt.imshow(sobel_y_filter2d, cmap='gray')
plt.title('Sobel Y (filter2D)')
plt.axis('off')

plt.subplot(164)
plt.imshow(sobel_x_custom, cmap='gray')
plt.title('Sobel X (Custom)')
plt.axis('off')

plt.subplot(165)
plt.imshow(sobel_y_custom, cmap='gray')
plt.title('Sobel Y (Custom)')
plt.axis('off')

plt.subplot(166)
plt.imshow(sobel_x_sep, cmap='gray')
plt.title('Sobel X (Separable)')
plt.axis('off')

#save plots
plt.tight_layout()
plt.savefig('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/sobel_filtering.png')
plt.show()