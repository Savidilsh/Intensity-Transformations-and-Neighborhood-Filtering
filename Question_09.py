import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img_orig = cv.imread('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/daisy.jpg', cv.IMREAD_COLOR)

# Create a mask and foreground, background models to initialize GrabCut algorithm
mask = np.zeros(img_orig.shape[:2], np.uint8)
foreground_model = np.zeros((1, 65), np.float64)
background_model = np.zeros((1, 65), np.float64)

rect = (50, 50, img_orig.shape[1] - 50, img_orig.shape[0] - 50)  # Define rectangles around the foreground

cv.grabCut(img_orig, mask, rect, background_model, foreground_model, 5, cv.GC_INIT_WITH_RECT)  # Apply Grabcut algorithm

new_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # Modify the mask

# Extract the foreground and background images
foreground_img = img_orig * new_mask[:, :, np.newaxis]
background_img = img_orig * (1 - new_mask[:, :, np.newaxis])

background_blurred_img = cv.GaussianBlur(background_img, (21, 21), 0)  # Apply Gaussian blur to the background

enhanced_img = foreground_img + background_blurred_img  

# Display final results
plt.figure(figsize=(20, 4))

plt.subplot(171)
plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(172)
plt.imshow(new_mask, cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')

plt.subplot(173)
plt.imshow(cv.cvtColor(foreground_img, cv.COLOR_BGR2RGB))
plt.title('Foreground Image')
plt.axis('off')

plt.subplot(174)
plt.imshow(cv.cvtColor(background_img, cv.COLOR_BGR2RGB))
plt.title('Background Image')
plt.axis('off')

plt.subplot(175)
plt.imshow(cv.cvtColor(background_blurred_img, cv.COLOR_BGR2RGB))
plt.title('Background Blurred Image')
plt.axis('off')

plt.subplot(176)
plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(177)
plt.imshow(cv.cvtColor(enhanced_img, cv.COLOR_BGR2RGB))
plt.title('Enhanced Image')
plt.axis('off')

plt.savefig('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/image_segmentation.png')
plt.show()
