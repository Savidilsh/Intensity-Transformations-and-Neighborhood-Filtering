import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Read input image
original_image = cv.imread('C:\\Users\\Savindu Dilshan\\Desktop\\Github\\Intensity-Transformations-and-Neighborhood-Filtering\\a1images\\jeniffer.jpg', cv.IMREAD_COLOR)

# Convert to HSV color space
hsv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
hue, saturation, value = cv.split(hsv_image)

# Create foreground mask from value channel
_, foreground_mask = cv.threshold(value, 100, 255, cv.THRESH_BINARY)

# Separate foreground
image_foreground = cv.bitwise_and(original_image, original_image, mask=foreground_mask)

# Process value channel for foreground
value_foreground = cv.bitwise_and(value, value, mask=foreground_mask)
histogram = np.bincount(value_foreground.ravel(), minlength=256)
cumulative_dist = np.cumsum(histogram)
normalized_dist = cumulative_dist * 255 / cumulative_dist[-1]

# Apply histogram equalization to foreground
equalized_value = np.interp(value_foreground, np.arange(256), normalized_dist).astype(np.uint8)

# Reconstruct equalized image
equalized_hsv = cv.merge([hue, saturation, equalized_value])
equalized_foreground = cv.cvtColor(equalized_hsv, cv.COLOR_HSV2BGR)

# Combine with background
image_background = cv.bitwise_and(original_image, original_image, mask=cv.bitwise_not(foreground_mask))
final_image = cv.add(equalized_foreground, image_background)

# Display results
plt.figure(figsize=(20, 4))
display_images = [
    (hue, 'Hue'),
    (saturation, 'Saturation'),
    (value, 'Value'),
    (foreground_mask, 'Mask'),
    (cv.cvtColor(original_image, cv.COLOR_BGR2RGB), 'Original'),
    (cv.cvtColor(final_image, cv.COLOR_BGR2RGB), 'Result')
]

for i, (img, title) in enumerate(display_images, 1):
    plt.subplot(1, 6, i)
    plt.imshow(img, cmap='gray' if i <= 4 else None)
    plt.title(title)
    plt.axis('off')

plt.show()
