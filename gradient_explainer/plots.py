# Run script with:
#python3 -m gradient_explainer.plots

import joblib

shap_values = joblib.load('gradient_explainer/shap_values.joblib')

shap.image_plot(shap_values, to_explain)#, indexes)
plt.savefig(f'gradient_explainer/images/image_plot_2.png', bbox_inches='tight', pad_inches=0)
plt.close()