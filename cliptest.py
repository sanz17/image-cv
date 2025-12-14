import torch
import clip
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open("test.jpg")).unsqueeze(0).to(device)

labels = ["cat", "dog", "car", "person", "pizza", "laptop"]
text_tokens = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).squeeze(0)


best_idx = similarity.argmax().item()
print("Predicted label:", labels[best_idx])
print("Similarity score:", similarity[best_idx].item())
