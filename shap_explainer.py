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

# Convert the image to the appropriate format for cv2
image_cv2 = (image * 255).astype(np.uint8)

# Save the image using cv2
cv2.imwrite('my_images/displayed_image_cv2.png', image_cv2)