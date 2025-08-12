import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load image
img_orig = cv.imread('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/highlights_and_shadows.jpg', cv.IMREAD_COLOR)

# Convert to L*a*b* color space
lab_img = cv.cvtColor(img_orig, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab_img)

# Apply gamma correction 
gamma = 0.7  
l_corrected = np.power(l / 255.0, gamma) * 255.0
l_corrected = l_corrected.astype(np.uint8)

# Merge back and convert to BGR
lab_corrected = cv.merge([l_corrected, a, b])
corrected_img = cv.cvtColor(lab_corrected, cv.COLOR_LAB2BGR)

# Compute histograms for original and corrected L planes
hist_orig = cv.calcHist([l], [0], None, [256], [0, 256])
hist_corrected = cv.calcHist([l_corrected], [0], None, [256], [0, 256])

# Display results using subplots similar to the example
plt.figure(figsize=(20, 4))

plt.subplot(141)
plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(142)
plt.imshow(cv.cvtColor(corrected_img, cv.COLOR_BGR2RGB))
plt.title(f'Corrected Image (γ = {gamma})')
plt.axis('off')

plt.subplot(143)
plt.plot(hist_orig)
plt.title('Original Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.subplot(144)
plt.plot(hist_corrected)
plt.title('Corrected Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

#save plots
plt.tight_layout()
plt.savefig('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/highlights_and_shadows_corrected.png')
plt.savefig('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/highlights_and_shadows_histograms.png')

plt.show()