import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load image
img_orig = cv.imread('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)

# Define intensity transformation for white matter 
lut_white = np.zeros(256, dtype=np.uint8)
for i in range(256):
    if i < 100:
        lut_white[i] = int(i * 255 / 100) 
    else:
        lut_white[i] = 255  

white_img = cv.LUT(img_orig, lut_white)

# Define intensity transformation for gray matter 
lut_gray = np.zeros(256, dtype=np.uint8)
for i in range(256):
    if 100 <= i < 200:
        lut_gray[i] = int((i - 100) * 255 / 100)  
    else:
        lut_gray[i] = 0  

gray_img = cv.LUT(img_orig, lut_gray)

# Display results using subplots
plt.figure(figsize=(20, 4))

plt.subplot(161)
plt.imshow(img_orig, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(162)
plt.imshow(white_img, cmap='gray')
plt.title('White Matter Accentuated')
plt.axis('off')

plt.subplot(163)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Matter Accentuated')
plt.axis('off')

plt.subplot(164)
plt.plot(lut_white)
plt.title('White Matter Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')

plt.subplot(165)
plt.plot(lut_gray)
plt.title('Gray Matter Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')

#save the transformed images
cv.imwrite('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice_white.jpg', white_img)
cv.imwrite('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/brain_proton_density_slice_gray.jpg', gray_img)
plt.tight_layout()  
plt.show()