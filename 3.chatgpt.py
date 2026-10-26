
import cv2
import numpy as np


def enhance_image(image):
    # 1. Reduce noise
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, 3, 3, 7, 21
    )

    # 2. Improve local contrast and visibility
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)

    contrast = cv2.cvtColor(
        cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR
    )

    # 3. Sharpen details
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    sharpened = cv2.addWeighted(
        contrast, 1.5, blurred, -0.5, 0
    )

    # 4. Slightly improve saturation
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = np.clip(s.astype(np.float32) * 1.12, 0, 255)
    s = s.astype(np.uint8)

    enhanced = cv2.cvtColor(
        cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR
    )

    return enhanced


# Give your image path here
image_path = r"F:\kpit\image-data-quality\sample1_foreground_background\512_896\n008-2018-08-27-11-48-51-0400__CAM_BACK_LEFT__1535385092297405.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Enhance the image
enhanced = enhance_image(image)

# Add labels
original_labeled = image.copy()
enhanced_labeled = enhanced.copy()

font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.rectangle(original_labeled, (10, 10), (180, 50), (0, 0, 0), -1)
# cv2.putText(original_labeled, "ORIGINAL", (20, 38),
#             font, 0.8, (255, 255, 255), 2)

# cv2.rectangle(enhanced_labeled, (10, 10), (200, 50), (0, 0, 0), -1)
# cv2.putText(enhanced_labeled, "ENHANCED", (20, 38),
#             font, 0.8, (255, 255, 255), 2)

# Stack vertically: original on top, enhanced below
comparison = np.vstack((original_labeled, enhanced_labeled))

# Display comparison
# cv2.imshow("Original vs Enhanced", comparison)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save enhanced image and comparison
cv2.imwrite("chatgpt_output/enhanced_image.jpg", enhanced)
cv2.imwrite("chatgpt_output/comparison.jpg", comparison)

print("Saved enhanced_image.jpg and comparison.jpg")
print("Original dimensions:", image.shape[1], "x", image.shape[0])
print("Enhanced dimensions:", enhanced.shape[1], "x", enhanced.shape[0])
