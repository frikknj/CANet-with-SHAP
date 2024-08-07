# Simple version of CANet with SHAP explanations

## Related work
See https://github.com/xmengli/CANet for the original code.

See CANet paper: [CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading](https://arxiv.org/abs/1911.01376)

## Preparation

Install CUDA and pythorch.

Download the [Messidor dataset](https://www.adcis.net/en/third-party/messidor/).

Save the data in the missdor folder.

## Train and test

Run the train and test file to train the model.

## Generate SHAP image plots

In the gradient_explainer.py and partition_explainer.py files see comments on how to run.
