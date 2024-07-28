# Run script with:
#python3 -m partition_explainer.partition_explainer

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

def preprocess_image(input_image):
    image_cropped = input_image[0:1488,376:1864]
    image_resized = cv2.resize(image_cropped, (224,224))
    return image_resized


def make_list_images(directory_path, amount_of_images):
    images_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            image_preprocessed = preprocess_image(image)
            images_list.append(image_preprocessed)
    return np.array(images_list[0: amount_of_images])


def f(X):
    tmp = X.copy()
    tmp = create_tensors(X)
    return model(tmp)


def generate_shap_values(images):
    masker = shap.maskers.Image("inpaint_telea", images[0].shape)

    class_names = ['Grade_0','Grade_1']

    explainer = shap.Explainer(f, masker, output_names=class_names)

    shap_values = explainer(images)

    return shap_values

def create_tensors(images):
    tensors = []
    for image in images:
        image_added_batch_dimension = np.expand_dims(image, axis=0)
        tensor = torch.Tensor(image_added_batch_dimension).permute(0, 3, 1, 2).float()
        tensors.append(tensor)
    return torch.cat(tensors)


directory_path = "missdor/Base11"

images = make_list_images(directory_path, 1)
shap_values = generate_shap_values(images)
print(type(shap_values))

joblib.dump(shap_values, 'partition_explainer/shap_values.joblib')

def make_plot2(shap_values):
    for i in range(len(shap_values)):
        shap.image_plot(shap_values[i])

        plt.savefig(f'partition_explainer/images/image_plot_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

def make_plot(shap_values):
    shap.image_plot(shap_values)

    plt.savefig(f'partition_explainer/images/image_plot_DETTE.png', bbox_inches='tight', pad_inches=0)
    plt.close()

make_plot(shap_values)