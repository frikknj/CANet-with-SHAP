# Run script with:
#python3 -m partition_explainer.plots
# Alternatively:
#python3 -m partition_explainer.partition_explainer; python3 -m partition_explainer.plots

import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

shap_values = joblib.load('partition_explainer/shap_values.joblib')

shap_values.data = shap_values.data.astype(np.float32)
shap_values.values = [np.array(val).astype(np.float32) for val in shap_values.values]

shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
)

plt.savefig(f'partition_explainer/images/image_plot.png', bbox_inches='tight', pad_inches=0)
plt.close()