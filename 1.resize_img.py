
import cv2
import os

# Input and output folders
input_folder = r"F:\kpit\image-data-quality\sample2"
output_folder = r"F:\kpit\image-data-quality\sample2\resized_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read image
        image = cv2.imread(input_path)

        if image is None:
            print(f"Could not read: {filename}")
            continue

        # Resize to width=256, height=448
        resized_image = cv2.resize(image, (448, 256))

        # Save resized image
        cv2.imwrite(output_path, resized_image)

        print(f"Resized: {filename}")

print("All images processed successfully!")
