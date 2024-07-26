# Run script with:
#python3 -m gradient_explainer.gradient_explainer

from config import Opt

opt = Opt()
opt.isTrain = False

from model import ClsNet
import torch
import numpy as np
import cv2
import os
import shap
import matplotlib.pyplot as plt
import json

model = ClsNet(opt)
model_dict = model.state_dict()
pretrained_dict = torch.load('checkpoint/mil_100.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def normalize(image):
    if image.max() > 1:
        image = image.astype(np.float32) / 255
    image = (image - mean) / std
    print(torch.tensor(image).float().shape)
    x = torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
    print(x.shape)
    print(torch.Tensor(image).permute(0, 3, 1, 2).float().shape)
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

def scale(image):
    image_cropped = image[0:1488,376:1864]
    image_resized = cv2.resize(image_cropped, (224,224))
    return image_resized

def make_list_images(directory_path):
    images_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            images_list.append(image)
    return images_list

def scale_all(images):
    images_scaled = []
    for image in images:
        image_scaled = scale(image)
        images_scaled.append(image_scaled)
    return images_scaled

directory_path = "missdor/Base11"

X = make_list_images(directory_path)
X = np.array(scale_all(X))

to_explain = X[[38,39,40]]

e = shap.GradientExplainer(model, normalize(X))
shap_values,indexes = e.shap_values(normalize(to_explain), ranked_outputs=2)

shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values] #torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
shap_values = [s.transpose(2,3,1,0) for s in shap_values]
shap_values = list(map(np.array, zip(*shap_values)))

def save_to_json_file(all_shap_values):
    listed_dictionaries = []
    for shap_value in all_shap_values:
        shap_values_dic = {
            "values": shap_value.tolist()
        }
        listed_dictionaries.append(shap_values_dic)

    with open('gradient_explainer/shap_values.json', "w") as json_file:
        json.dump(listed_dictionaries, json_file, indent=4)

save_to_json_file(shap_values)

shap.image_plot(shap_values, to_explain)#, indexes)
plt.savefig(f'gradient_explainer/images/image_plot_2.png', bbox_inches='tight', pad_inches=0)
plt.close()