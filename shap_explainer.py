from shap_base import preprocess_images, directory_path, model, save_to_json_file, plots

import shap
import torch

images = preprocess_images(directory_path, 2)

def generate_shap_values(model, images):
    shap_values_list = []
    # Create a masker
    masker = shap.maskers.Image("blur(128,128)", images[1].shape[1:])
    # Initialize the explainer with the model and masker
    explainer = shap.Explainer(model, masker=masker) # outputs=shap.Explanation.argsort.flip[:4]
    for image in images:
        preprocessed_img = torch.tensor(image) #.float()
        shap_values = explainer(preprocessed_img)
        shap_values_list.append(shap_values)
    return shap_values_list

shap_values_list = generate_shap_values(model, images)
#save_to_json_file(shap_values_list[0])
plots(shap_values_list, images, "normal_explainer_images")