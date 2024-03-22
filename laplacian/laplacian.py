import numpy as np
import cv2

# Load the image
image_path = 'C:/Users/swani/OneDrive/Desktop/sem4/filter_dsp/stone.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define Laplacian kernels
laplacian_kernel1 = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

laplacian_kernel2 = np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]])

# Apply Laplacian filters
filtered_image1 = cv2.filter2D(gray_image, -1, laplacian_kernel1)
filtered_image2 = cv2.filter2D(gray_image, -1, laplacian_kernel2)

# Calculate the second order derivative for each Laplacian filter
delta2_image1 = np.abs(filtered_image1 - gray_image)
delta2_image2 = np.abs(filtered_image2 - gray_image)

# Sharpen the images by subtracting the second derivative
sharpened_image1 = np.clip(gray_image - delta2_image1, 0, 255).astype(np.uint8)
sharpened_image2 = np.clip(gray_image - delta2_image2, 0, 255).astype(np.uint8)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image 1', filtered_image1)
cv2.imshow('Filtered Image 2', filtered_image2)
cv2.imshow('Sharpened Image 1', sharpened_image1)
cv2.imshow('Sharpened Image 2', sharpened_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
