import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Load the image 
img_orig = cv.imread('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/spider.png', cv.IMREAD_COLOR)

# Split into HSV planes
hsv_img = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv_img)

# Apply the intensity transformation to saturation plane
a = 0.3 
sigma = 70
transformation = lambda x: np.minimum(x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2)), 255)
s_transformed = transformation(s).astype(np.uint8)

# Recombine the planes
hsv_enhanced = cv.merge([h, s_transformed, v])
enhanced_img = cv.cvtColor(hsv_enhanced, cv.COLOR_HSV2BGR)

# Plot the intensity transformation
x = np.arange(256)
trans_plot = transformation(x)

# Display results using subplots
plt.figure(figsize=(20, 4))

plt.subplot(161)
plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(162)
plt.imshow(h, cmap='gray')
plt.title('Hue Plane')
plt.axis('off')

plt.subplot(163)
plt.imshow(s, cmap='gray')
plt.title('Saturation Plane')
plt.axis('off')

plt.subplot(164)
plt.imshow(v, cmap='gray')
plt.title('Value Plane')
plt.axis('off')

plt.subplot(165)
plt.imshow(cv.cvtColor(enhanced_img, cv.COLOR_BGR2RGB))
plt.title(f'Vibrance-Enhanced Image (a = {a})')
plt.axis('off')

plt.subplot(166)
plt.plot(x, trans_plot)
plt.title('Intensity Transformation')
plt.xlabel('Input Saturation')
plt.ylabel('Output Saturation')

#save plots
plt.tight_layout()
plt.savefig('C:/Users/Savindu Dilshan/Desktop/Github/Intensity-Transformations-and-Neighborhood-Filtering/a1images/enhanced.png')
plt.show()
