# Run script with:
#python3 -m partition_explainer.partition_explainer3

from config import Opt

opt = Opt()
opt.isTrain = False

from model import ClsNet
import torch
import torchvision
import numpy as np
import cv2
import os
import shap
import matplotlib.pyplot as plt
import joblib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def scale_all(images):
    images_scaled = []
    for image in images:
        image_scaled = scale(image)
        images_scaled.append(image_scaled)
    return images_scaled

def make_list_images(directory_path):
    images_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            images_list.append(image)
    return images_list

directory_path = "missdor/Base11"

X = make_list_images(directory_path)
X = np.array(scale_all(X))

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x

transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
    torchvision.transforms.Normalize(mean=mean, std=std),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)

def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img))
    output = model(img)
    return output

Xtr = transform(torch.Tensor(X))
masker_blur = shap.maskers.Image("blur(128,128)", Xtr[0].shape)

class_names = ['Grade_0','Grade_1']
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

shap_values = explainer(Xtr[0:1], max_evals=10)

print(shap_values.data.shape, shap_values.values.shape)

shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
    labels=shap_values.output_names,
)

plt.savefig(f'partition_explainer/images/image_plot_FINAL.png', bbox_inches='tight', pad_inches=0)
plt.close()