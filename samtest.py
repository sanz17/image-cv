import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

# 1. Load SAM Model
sam_checkpoint = "/Users/sanhitakundu/Desktop/codebanks/image query/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](sam_checkpoint)
predictor = SamPredictor(sam)

# 2. Load image
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# 3. Choose a point manually
point = np.array([[300, 200]])  # (x, y)
label = np.array([1])  # 1=foreground

# 4. Run SAM prediction
masks, scores, logits = predictor.predict(
    point_coords=point,
    point_labels=label,
    multimask_output=False,
)

mask = masks[0]
overlay = image.copy()
overlay[mask] = (255, 0, 0)  # blue mask

alpha = 0.6
result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# Draw the point
cv2.circle(result, (point[0][0], point[0][1]), 5, (0, 255, 0), -1)

# 6. Show result
cv2.imshow("SAM Segmentation", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Mask shape:", mask.shape)
