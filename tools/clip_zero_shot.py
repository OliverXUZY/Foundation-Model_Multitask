import os
import clip
import torch
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder


# download mini-imagenet here https://drive.google.com/drive/folders/17a09kkqVivZQFggCw9I_YboJ23tcexNM
data_root = "/srv/home/zxu444/datasets/mini-imagenet/test"
model_name = 'ViT-B/32'
n_way = 5

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device)
print("Finish Load Model", model_name)

with open('classnames.txt') as f:
    lines = [line.rstrip() for line in f]

class_to_name = {}
for line in lines:
    s_id = line.find(' ')
    class_to_name[line[:s_id]] = line[s_id+1:]
    
test_dataset = ImageFolder(root = data_root, transform = preprocess)
idx_to_name = {}
for c in test_dataset.class_to_idx:
    idx_to_name[test_dataset.class_to_idx[c]] = class_to_name[c]
n_class = len(idx_to_name)
print("Finish Load Data")

correct = 0
for step in range(len(test_dataset)):
    if step % 100 == 0:
        print(f"{step}/{len(test_dataset)}: correct {correct}")
    # Prepare the inputs
    image, class_id = test_dataset[step]
    image_input = image.unsqueeze(0).to(device)
    class_ids = np.random.choice(n_class, n_way, replace=False)
    while class_id in class_ids:
        class_ids = np.random.choice(n_class, n_way, replace=False)
    class_ids[0] = class_id    # the first-index is ground-truth
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {idx_to_name[j]}") for j in class_ids]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(n_way)
    if indices[0] == 0:
        correct += 1

print(correct / len(test_dataset))
