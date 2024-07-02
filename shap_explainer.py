import shap
from model import ClsNet
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import os

from config import Opt
opt = Opt()

directory_path = "missdor/Base11"
image_path = "missdor/Base11/20051019_38557_0100_PP.tif"
image = cv2.imread(image_path)

image_resized = cv2.resize(image, (224, 224))

# Add batch dimension
image_resized = np.expand_dims(image_resized, axis=0)

#cv2.imwrite('my_images/image.png', image)

model = ClsNet(opt)

def preprocess_images(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".tif"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image, (224,224)) # Disse må samsvare med pixel verdiene oppgitt i artikkel.
            image_resized = np.expand_dims(image_resized, axis=0)
            images.append(image_resized)
    return images[0:5]

images = preprocess_images(directory_path)

def generate_shap_values(model, images):
    shap_values_list = []
    # Create a masker
    masker = shap.maskers.Image("blur(128,128)", images[1].shape[1:])
    # Initialize the explainer with the model and masker
    explainer = shap.Explainer(model, masker=masker) # outputs=shap.Explanation.argsort.flip[:4]
    for image in images:
        preprocessed_img = torch.from_numpy(image) #.float()
        shap_values = explainer(preprocessed_img)
        shap_values_list.append(shap_values)
    return shap_values_list

shap_values_list = generate_shap_values(model, images)
print(shap_values_list[0].shape)

def generate_shap_values_deep_explainer(model, images, background_data):
    shap_values_list = []
    explainer = shap.DeepExplainer(model, background_data)
    for image in images:
        preprocessed_img = torch.from_numpy(image)
        shap_values = explainer(preprocessed_img)
        shap_values_list.append(shap_values)
    return shap_values_list

#shap_values_list_deep_explainer = generate_shap_values_deep_explainer(model, images, images[0])

def save_to_json_file(values): #Saving shap values to json file.
    filepath = "my_images/shap_values.json"

    shap_values_data = {
    "values": values.values.tolist(),  # SHAP values
    "base_values": values.base_values.tolist(),  # Base values
    "data": values.data.tolist()  # Input data
    }

    with open(filepath, "w") as json_file:
        json.dump(shap_values_data, json_file, indent=4)

save_to_json_file(shap_values_list[0])

def plots(shap_values_list, background_images):
    for i, (shap_values, background_image) in enumerate(zip(shap_values_list, background_images)):
        # Convert the SHAP values to a format suitable for plotting
        shap_values_np = [sv.values for sv in shap_values]
        shap_values_np = np.array(shap_values_np)

        # Plot the SHAP values
        shap.image_plot(shap_values_np, background_image)

        # Save the plot to a file
        plt.savefig(f'my_images/shap_plot_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    background_image_to_save = (background_image * 255).astype(np.uint8)
    background_image_to_save = np.squeeze(background_image_to_save)
    cv2.imwrite(f'my_images/image_resized.png', background_image_to_save)

plots(shap_values_list, images)