import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load image
img_orig = cv.imread('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)

# Define intensity transformation for white matter 
lut_white = np.zeros(256, dtype=np.uint8)
lut_white[0:175] = np.linspace(255, 255, 175)
lut_white[175:200] = np.linspace(255, 0, 25)
lut_white[200:] = 0

# Apply transformation for gray matter
lut_gray = np.zeros(256, dtype=np.uint8)
lut_gray[0:50] = 255
lut_gray[50:125] = np.linspace(255, 0, 75)
lut_gray[125:200] = np.linspace(0, 255, 75)
lut_gray[200:] = 255

white_img = cv.LUT(img_orig, lut_white)
gray_img = cv.LUT(img_orig, lut_gray)

# Display results using subplots
plt.figure(figsize=(20, 4))

plt.subplot(151)
plt.imshow(img_orig, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(152)
plt.imshow(white_img, cmap='gray')
plt.title('White Matter Accentuated')
plt.axis('off')

plt.subplot(153)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Matter Accentuated')
plt.axis('off')

plt.subplot(154)
plt.plot(lut_white)
plt.title('White Matter Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')

plt.subplot(155)
plt.plot(lut_gray)
plt.title('Gray Matter Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')

#save the transformed images
cv.imwrite('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice_white.jpg', white_img)
cv.imwrite('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice_gray.jpg', gray_img)
plt.tight_layout()  
plt.show()