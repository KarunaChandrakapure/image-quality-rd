import cv2
import numpy as np
def apply_vibrance(image: np.ndarray, amount: float) -> np.ndarray:
    """
    An Adobe-style 'Vibrance' algorithm.
    
    Unlike Saturation, Vibrance:
    1. Boosts low-saturation pixels more aggressively.
    2. Protects highly saturated pixels from clipping.
    
    Args:
        image: Input BGR image (uint8).
        amount: Strength of the effect. 
                0.0 = no change.
                1.0 = moderate boost.
                2.0+ = very strong 'pop'.
    """
    # 1. Convert to HSV to isolate the Saturation (S) channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    # 2. Normalize Saturation to a 0.0 - 1.0 range for math
    s_norm = s / 255.0
    # 3. The "Protection" Factor (The secret sauce)
    # We want a multiplier that is HIGH when saturation is LOW,
    # and LOW when saturation is HIGH.
    # formula: (1 - saturation_level)
    protection_factor = 1.0 - s_norm
    # 4. Calculate the Boost
    # We scale the 'amount' by the protection factor.
    # This ensures that if S is 0.9 (90% saturated), the boost is tiny.
    # If S is 0.1 (10% saturated), the boost is large.
    boost = amount * protection_factor
    # 5. Apply the boost to the saturation channel
    # We use an additive approach for a smoother, more natural transition
    s_new = s_norm + boost
    
    # 6. Clip and convert back to 0-255
    s_new = np.clip(s_new, 0.0, 1.0) * 255.0
    
    # 7. Reconstruct the image
    hsv_new = cv2.merge([h, s_new, v])
    result = cv2.cvtColor(hsv_new.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result
def apply_advanced_vibrance(image: np.ndarray, amount: float, use_clahe: bool = True) -> np.ndarray:
    """
    A complete 'Pro' pipeline: 
    Vibrance (for color pop) + CLAHE (for local contrast/brightness uniformity).
    """
    # Step 1: Color Pop via Vibrance
    vibrant_img = apply_vibrance(image, amount)
    # Step 2: Contrast/Brightness Uniformity via CLAHE
    # This fixes the 'whitewashed' or 'dark' look without changing colors.
    hsv = cv2.cvtColor(vibrant_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if use_clahe:
        # CLAHE works on 8-bit images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # We ensure 'v' is uint8 before passing to CLAHE, 
        # then convert back to float32 just to be safe for merging logic
        v = clahe.apply(v.astype(np.uint8)).astype(np.float32)
    else:
        v = v.astype(np.float32)
    # IMPORTANT FIX: Ensure h and s are also float32 so they match 'v'
    # or convert everything to uint8. Let's convert everything to uint8
    # as it is more standard for final image display/saving.
    h = h.astype(np.float32)
    s = s.astype(np.float32)
    # Now all channels are float32, so merge will work
    hsv_final = cv2.merge([h, s, v])
    
    # Convert the merged float32 HSV back to uint8 BGR
    # We clip to 0-255 first to prevent wrap-around artifacts
    result = cv2.cvtColor(np.clip(hsv_final, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result

# -------------------------------------------------------------------------
# Test/Demo Routine
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Load your image
    img = cv2.imread('argo_waymo_scenes/argo/scene_1_argo_val_1/ring_rear_left/0.1.png')
    
    if img is None:
        print("Error: Could not load image.")
    else:
        # TEST 1: Standard Saturation (The "Bad" way - causes clipping)
        # We simulate this by just multiplying
        hsv_bad = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_bad[:,:,1] = np.clip(hsv_bad[:,:,1] * 1.5, 0, 255)
        bad_sat = cv2.cvtColor(hsv_bad.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # TEST 2: Adobe-style Vibrance (The "Good" way)
        # amount=1.0 is a safe starting point.
        good_vibrance = apply_advanced_vibrance(img, amount=1.2, use_clahe=True)
        # Display results
        # We stack them for comparison
        # [Original | Bad Saturation | Adobe Vibrance]
        comparison = np.hstack([img, bad_sat, good_vibrance])
        
        # Resize for viewing if image is too large
        scale_percent = 50 
        width = int(comparison.shape[1] * scale_percent / 100)
        height = int(comparison.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        resized_comp = cv2.resize(comparison, dim, interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Comparison: Original | Bad Saturation | Adobe Vibrance', resized_comp)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
