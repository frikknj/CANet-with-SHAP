import shap
from model import ClsNet
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img

from mildataset import IDRid_Dataset

from config import Opt
opt = Opt()

# Create an instance of your model
model = ClsNet(opt)  # Assuming you have an 'opt' object for initialization

# Create a masker
masker = shap.maskers.Image("inpaint_telea", (3, 224, 224))  # Adjust shape according to your input

# Initialize the explainer with the model and masker
explainer = shap.Explainer(model, masker=masker)

# Assuming you have some image data to explain
image_path = "missdor/Base11/20051019_38557_0100_PP.tif"
image = img.imread(image_path)

# Display the image and save it to a file
plt.imshow(image)
plt.axis('off')  # Hide axes for a cleaner display
plt.savefig('my_images/displayed_image.png', bbox_inches='tight', pad_inches=0)

# Get SHAP values
#image_data = np.random.randn(1, 3, 224, 224)  # Replace with actual image data
#shap_values = explainer(image_data)
