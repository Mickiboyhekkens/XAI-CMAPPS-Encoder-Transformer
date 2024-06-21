import torch
import numpy as np
import torch.nn as nn
from lime.lime_tabular import LimeTabularExplainer

# Ensure device is set (assuming you're using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = torch.load('1_model_final.pt', map_location=device)
best_model.to(device)
best_model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, X_input):
        #X_input = X_input.detach().numpy()
        #X_input = X_input.detach().numpy()
        test_dataset = supDataset(data_list=[X_input], targets=[np.array([0])])
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_superv,
                                     batch_size=1,
                                     shuffle=False,
                                     # num_workers=2,
                                     pin_memory=True)
        
        for batch in test_dataloader:
            X, targets, padding_masks = batch[0], batch[1], batch[2]
            # move data to GPU/ any available device
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)
            
            print(X.shape, targets.shape, padding_masks.shape)
                                      
        return self.model(X, padding_masks, transform='linear')

# Instantiate the wrapped model
wrapper = ModelWrapper(best_model)

# Flatten the input data for LIME
X_input = x_test_list[0].flatten()

# Define the LimeTabularExplainer
explainer = LimeTabularExplainer(X_numpy, mode='regression', verbose=True)
# Explain a single instance
exp = explainer.explain_instance(X_input[0], wrapper.predict, num_features=10)

# Visualize the explanation
exp.show_in_notebook(show_all=False)

# Ensure device is set (assuming you're using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = torch.load('1_model_final.pt', map_location=device)
best_model.to(device)
best_model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, X_input):
        #X_input = X_input.detach().numpy()
        #X_input = X_input.detach().numpy()
        test_dataset = supDataset(data_list=[X_input], targets=[np.array([0])])
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_superv,
                                     batch_size=1,
                                     shuffle=False,
                                     # num_workers=2,
                                     pin_memory=True)
        
        for batch in test_dataloader:
            X, targets, padding_masks = batch[0], batch[1], batch[2]
            # move data to GPU/ any available device
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)
            
            print(X.shape, targets.shape, padding_masks.shape)
                                      
        return self.model(X, padding_masks, transform='linear')

# Instantiate the wrapped model
wrapper = ModelWrapper(best_model)

X = x_test_list[2]
X = torch.tensor(X).to(device)
# Make predictions to verify the model works
predictions = wrapper(X)
print(predictions)

# Initialize a SHAP explainer with the validation data
explainer = shap.DeepExplainer(wrapper, X)
shap_values = explainer.shap_values(X)
# init the JS visualization code
shap.initjs()

print(explainer.expected_value)

print(shap_values)
is_all_zero = np.all(shap_values == 0)
print(is_all_zero)

import torch
import numpy as np
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import torch.nn as nn

# Ensure device is set (assuming you're using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = torch.load('1_model_final.pt', map_location=device)
best_model.to(device)
best_model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, X_input):
        #X_input = X_input.detach().numpy()
        X_input = X_input.detach().numpy()
        test_dataset = supDataset(data_list=[X_input], targets=[np.array([0])])
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_superv,
                                     batch_size=1,
                                     shuffle=False,
                                     # num_workers=2,
                                     pin_memory=True)
        
        for batch in test_dataloader:
            X, targets, padding_masks = batch[0], batch[1], batch[2]
            # move data to GPU/ any available device
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)
            
            print(X.shape, targets.shape, padding_masks.shape)
                                      
        return self.model(X, padding_masks, transform='linear')

wrapper = ModelWrapper(best_model)

# Instantiate Integrated Gradients
ig = IntegratedGradients(wrapper)

# Select an instance to explain)
X_input = torch.tensor(x_test_list[2])  # Select the first instance from the batch
X_input = X_input.clone().detach().requires_grad_(True)  # Ensure it requires gradients
X_input = X_input
print(X_input.shape)

y = wrapper(X_input)
print(y)

# Define a baseline (e.g., zero tensor of the same shape as X_input)
X_zeros = torch.zeros_like(X_input)
print(X_zeros.shape)

# Calculate attributions
attributions, delta = ig.attribute(X_input, baselines=X_zeros, return_convergence_delta=True)

# Visualize the attributions
attributions_np = attributions.cpu().detach().numpy().reshape(-1, 362, 14)
viz.visualize_image_attr_multiple(
    [attributions_np],
    [X_input.cpu().detach().numpy().reshape(-1, 362, 14)],
    ["heat_map"],
    ["all"],
    titles=["Integrated Gradients"]
)
import torch
import numpy as np
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import torch.nn as nn

# Ensure device is set (assuming you're using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = torch.load('1_model_final.pt', map_location=device)
best_model.to(device)
best_model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, X_input):
        #X_input = X_input.detach().numpy()
        X_input = X_input.detach().numpy()
        test_dataset = supDataset(data_list=[X_input], targets=[np.array([0])])
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_superv,
                                     batch_size=1,
                                     shuffle=False,
                                     # num_workers=2,
                                     pin_memory=True)
        
        for batch in test_dataloader:
            X, targets, padding_masks = batch[0], batch[1], batch[2]
            # move data to GPU/ any available device
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)
            
            print(X.shape, targets.shape, padding_masks.shape)
                                      
        return self.model(X, padding_masks, transform='linear')

wrapper = ModelWrapper(best_model)

# Instantiate Integrated Gradients
ig = IntegratedGradients(wrapper)

# Select an instance to explain)
X_input = torch.tensor(x_test_list[2])  # Select the first instance from the batch
X_input = X_input.clone().detach().requires_grad_(True)  # Ensure it requires gradients
X_input = X_input
print(X_input.shape)

y = wrapper(X_input)
print(y)

# Define a baseline (e.g., zero tensor of the same shape as X_input)
X_zeros = torch.zeros_like(X_input)
print(X_zeros.shape)

# Calculate attributions
attributions, delta = ig.attribute(X_input, baselines=X_zeros, return_convergence_delta=True)

# Visualize the attributions
attributions_np = attributions.cpu().detach().numpy().reshape(-1, 362, 14)
viz.visualize_image_attr_multiple(
    [attributions_np],
    [X_input.cpu().detach().numpy().reshape(-1, 362, 14)],
    ["heat_map"],
    ["all"],
    titles=["Integrated Gradients"]
)

import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

# Ensure device is set (assuming you're using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = torch.load('1_model_final.pt', map_location=device)
best_model.to(device)
best_model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, X_input):
        #X_input = X_input.detach().numpy()
        test_dataset = supDataset(data_list=[X_input], targets=[np.array([0])])
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_superv,
                                     batch_size=1,
                                     shuffle=False,
                                     # num_workers=2,
                                     pin_memory=True)
        
        for batch in test_dataloader:
            X, targets, padding_masks = batch[0], batch[1], batch[2]
            # move data to GPU/ any available device
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)
            
            print(X.shape, targets.shape, padding_masks.shape)
                                      
        return self.model(X, padding_masks, transform='linear')

wrapper = ModelWrapper(best_model)


X_numpy = x_test_list[0]

explainer = shap.Explainer(wrapper, X_numpy, max_evals=12000)
shap_values = explainer(X_numpy)

import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

# Ensure device is set (assuming you're using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
best_model = torch.load('1_model_final.pt', map_location=device)
best_model.to(device)
best_model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, X_input):
        #X_input = X_input.detach().numpy()
        test_dataset = supDataset(data_list=[X_input], targets=[np.array([0])])
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_superv,
                                     batch_size=1,
                                     shuffle=False,
                                     # num_workers=2,
                                     pin_memory=True)
        
        for batch in test_dataloader:
            X, targets, padding_masks = batch[0], batch[1], batch[2]
            # move data to GPU/ any available device
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)
            
            print(X.shape, targets.shape, padding_masks.shape)
                                      
        return self.model(X, padding_masks, transform='linear')

wrapper = ModelWrapper(best_model)


X_numpy = x_test_list[0]

explainer = shap.Explainer(wrapper, X_numpy, max_evals=12000)
shap_values = explainer(X_numpy)

import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

# Ensure device is set (assuming you're using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the model
best_model = torch.load('1_model_final.pt', map_location=device)
best_model.to(device)

# Define a wrapper for the forward method
class ModelWrapper(nn.Module):
    def __init__(self, model, transform, masks):
        super(ModelWrapper, self).__init__()  # Initialize the parent class
        self.model = model
        self.transform = transform
        self.padding_masks = masks

    def forward(self, X):
        X_tensor = torch.tensor(X).to(device)
        lengths = []
        for x in X_tensor.numpy():
            i = 0
            for timepoint in x:
                if np.all(timepoint == 0):
                    break
                i += 1
            lengths.append(i)

        self.padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=362)
        self.padding_masks = self.padding_masks.to(device)
        return self.model(X_tensor, self.padding_masks, self.transform)

best_model.eval()

# Assuming val_dataloader and test_dataloader are defined
for batch in val_dataloader:
    X_val, targets_val, masks_val = batch[0], batch[1], batch[2]
    print(X_val.shape, targets_val.shape, masks_val.shape)
    break  # Just take the first batch for simplicity
    
for batch in test_dataloader:
    X, targets, padding_masks = batch[0], batch[1], batch[2]
    print(X.shape, targets.shape, padding_masks.shape)
    break  # Just take the first batch for simplicity
    

# Move data to GPU/ any available device
X = X.to(device)
targets = targets.to(device)
padding_masks = padding_masks.to(device)

# Instantiate the wrapped model
wrapper = ModelWrapper(best_model, 'linear', padding_masks)
# Make predictions to verify the model works
predictions = wrapper(X)
print(predictions)

wrapper.padding_masks = padding_masks
# Initialize a SHAP explainer with the validation data

explainer = shap.DeepExplainer(wrapper, X)

wrapper.padding_masks = padding_masks
X_val.requires_grad = True
X.requires_grad = True

X = X[0].unsqueeze(0)

shap_values = explainer.shap_values(X)
# init the JS visualization code
shap.initjs()

print(explainer.expected_value)

print(shap_values.shape)
is_all_zero = np.all(shap_values == 0)
print(is_all_zero)
print(shap_values[0].shape)
print(shap_values[0])
print(shap_values[0][0].shape)

# # Initialize the SHAP JS visualization
# shap.initjs()
# # Plot the Shapley values for the instance
# i = 0
# j = 0
# features = range(14)
# X = X.detach().numpy()
# print(shap_values[i][j])
# x_test_df = pd.DataFrame(data=X[i][j].reshape(1,14), columns = features)
# print(explainer.expected_value.shape)

# shap.force_plot(explainer.expected_value[0], shap_values[i][j], x_test_df)

# # Plot SHAP for ONLY one observation i
# x_test_df = pd.DataFrame(data=X[i], columns = features)
# expected = y_scaler.inverse_transform(explainer.expected_value.reshape(-1, 1))[0]
# shap.force_plot(expected, shap_values[i], x_test_df)
# ## Problem:  Can not take into account many observations at the same time.
# ### The pic below explain for only 1 observation of 20 time steps, each time step has 10 features.