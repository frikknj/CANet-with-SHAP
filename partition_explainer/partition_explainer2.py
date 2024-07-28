# Run script with:
#python3 -m partition_explainer.partition_explainer2

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
import joblib

model = ClsNet(opt)
model_dict = model.state_dict()
pretrained_dict = torch.load('checkpoint/mil_100.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model.eval()

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

def f(X):
    tmp = X.copy()
    tmp = normalize(X)
    return model(tmp)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def normalize(image):
    if image.max() > 1:
        image = image.astype(np.float32) / 255
    image = (image - mean) / std
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

directory_path = "missdor/Base11"

X = make_list_images(directory_path)
X = np.array(scale_all(X))

to_explain = X[[38,39]]

def generate_shap_values(images):
    masker = shap.maskers.Image("inpaint_telea", images[0].shape)

    class_names = ['Grade_0','Grade_1']

    explainer = shap.Explainer(f, masker, output_names=class_names)

    shap_values = explainer(to_explain, max_evals=100)

    return shap_values

shap_values = generate_shap_values(X)

def make_plot(shap_values):
    shap.image_plot(shap_values)

    plt.savefig(f'partition_explainer/images/image_plot_DETTE.png', bbox_inches='tight', pad_inches=0)
    plt.close()

shap_values = [s for s in shap_values] #Transform to list
print(type(shap_values[0]))
print(shap_values[0].shape)
make_plot(shap_values)

#Correct format: (1, 224, 224, 3)