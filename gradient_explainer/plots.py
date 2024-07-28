# Run script with:
#python3 -m gradient_explainer.plots
# Alternatively:
#python3 -m gradient_explainer.gradient_explainer; python3 -m gradient_explainer.plots

import shap
import joblib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

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
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_list.append(image_rgb)
    return images_list

directory_path = "missdor/Base11"

X = make_list_images(directory_path)
X = np.array(scale_all(X))

to_explain = X[[19, 31, 41]]

shap_values = joblib.load('gradient_explainer/shap_values.joblib')

shap.image_plot(shap_values, to_explain)
plt.savefig(f'gradient_explainer/images/image_plot.png', bbox_inches='tight', pad_inches=0)
plt.close()