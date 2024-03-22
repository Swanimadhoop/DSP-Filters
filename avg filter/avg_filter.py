import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'C:/Users/swani/OneDrive/Desktop/filter_dsp/noise.png'  # Full path to the image file

image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

    def calculate_psnr(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Original Image')
    plt.show()

    for size in kernel_sizes:
        filtered_image = cv2.blur(image_rgb, (size, size))
        psnr_value = calculate_psnr(image_rgb, filtered_image)

        plt.figure(figsize=(8, 6))
        plt.imshow(filtered_image)
        plt.axis('off')
        plt.title(f'Filtered Image (Kernel Size: {size}x{size}, PSNR: {psnr_value:.2f})')
        plt.show()

    psnr_values = [(size, calculate_psnr(image_rgb, cv2.blur(image_rgb, (size, size)))) for size in kernel_sizes]

    psnr_table = "| Kernel Size | PSNR |\n|-------------|------|\n"
    for size, psnr in psnr_values:
        psnr_table += f"| {size}x{size:<10} | {psnr:.2f} |\n"

    print(psnr_table)  # Print the Markdown table as plain text