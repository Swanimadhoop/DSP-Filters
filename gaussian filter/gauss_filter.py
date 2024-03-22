from IPython.display import Markdown
import cv2
import numpy as np

# Load the image from the specified path
image_path = 'C:/Users/swani/OneDrive/Desktop/filter_dsp/noise.png'
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the original image
    cv2.imshow("Original Image", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define kernel sizes for Gaussian blur
    kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

    # Function to calculate PSNR
    def calculate_psnr(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    # Lists to store filtered images and PSNR values
    filtered_images = []
    psnr_values = []

    # Apply Gaussian blur with different kernel sizes
    for size in kernel_sizes:
        filtered_image = cv2.GaussianBlur(image_rgb, (size, size), 0)
        psnr_value = calculate_psnr(image_rgb, filtered_image)

        psnr_values.append((size, psnr_value))
        filtered_images.append(filtered_image)

        # Display the filtered image
        cv2.imshow(f"Filtered Image (Kernel Size: {size}x{size})", filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Generate PSNR table as Markdown
    psnr_table = "| Kernel Size | PSNR |\n|-------------|------|\n"
    for size, psnr in psnr_values:
        psnr_table += f"| {size}x{size} | {psnr:.2f} |\n"

    # Print the PSNR table
    print(psnr_table)
