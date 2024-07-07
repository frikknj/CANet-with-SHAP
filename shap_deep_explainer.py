from shap_base import preprocess_images, directory_path, model, save_to_json_file, plots

import shap
import torch

images = preprocess_images(directory_path, 2)

def generate_shap_values_deep_explainer(model, images, background_data):
    shap_values_list = []
    explainer = shap.DeepExplainer(model, background_data)
    for image in images:
        preprocessed_img = torch.Tensor(image)
        shap_values = explainer(preprocessed_img)
        shap_values_list.append(shap_values)
    return shap_values_list

background_data = torch.from_numpy(images[0])

shap_values_list_deep_explainer = generate_shap_values_deep_explainer(model, images, background_data)
#save_to_json_file(shap_values_list[0]) # Saves shap values to json file.
plots(shap_values_list, images, "deep_explainer_images")