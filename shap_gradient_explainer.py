from shap_base import preprocess_images, directory_path, model, save_to_json_file, plots

import shap
import torch
import numpy as np

images = preprocess_images(directory_path, 1)

def generate_shap_values_gradient_explainer(model, images, background_data):
    shap_values_list = []
    explainer = shap.GradientExplainer(model, background_data)
    n = 0
    for image in images:
        preprocessed_img = torch.Tensor(image).permute(0, 3, 1, 2).float()
        shap_values = explainer(preprocessed_img)
        shap_values_list.append(shap_values)
        n += 1
        print(f"{n}/{len(images)}")
    return shap_values_list

background_data = torch.from_numpy(images[0]).permute(0, 3, 1, 2).float()

shap_values_list_gradient_explainer = generate_shap_values_gradient_explainer(model, images, background_data)
#save_to_json_file(shap_values_list[0]) # Saves shap values to json file.

def reshape(values):
    new_values = []
    for value in values:
        tensor = value.data
        permuted_tensor = tensor.permute(0, 3, 2, 1)
        new_value = shap.Explanation(values=permuted_tensor)
        new_values.append(new_value)
    return new_values

shap_values_list_gradient_explainer = reshape(shap_values_list_gradient_explainer)

plots(shap_values_list_gradient_explainer, images, "gradient_explainer_images")