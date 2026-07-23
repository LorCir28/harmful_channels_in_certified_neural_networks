import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.model_wrappers import ShiIBPModelWrapper
from ctrain.data_loaders import load_mnist
from ctrain.model_definitions.models_shi import CNN7_Shi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskedReLU(nn.Module):
    def __init__(self, relu, num_neurons, is_conv):
        super().__init__()
        self.relu = relu
        shape = (1, num_neurons, 1, 1) if is_conv else (1, num_neurons)
        self.mask = nn.Parameter(torch.ones(shape, device=device))
        self.mask.requires_grad = False
        
    def forward(self, x):
        return self.relu(x) * self.mask

def main():
    EPSILON = 0.3
    MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'mnist', 'cnn7_mnist_mtl_ibp_3.pt')

    print("Loading MNIST data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    _, test_loader_full = load_mnist(batch_size=128, val_split=False, data_root=data_root)

    # Use the first 1,000 images from the test set (indices 0 to 1000) for identification
    test_ds = test_loader_full.dataset
    ident_ds = Subset(test_ds, range(0, 1000))
    ident_loader = DataLoader(ident_ds, batch_size=128, shuffle=False)
    
    if hasattr(test_loader_full, 'normalised'):
        ident_loader.normalised = test_loader_full.normalised
        ident_loader.mean = getattr(test_loader_full, 'mean', None)
        ident_loader.std = test_loader_full.std
        ident_loader.min = test_loader_full.min
        ident_loader.max = test_loader_full.max
    
    print(f"Loading model weights from {MODEL_WEIGHTS_PATH}...")
    temp_model = CNN7_Shi(in_shape=(1, 28, 28), n_classes=10).to(device)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        # Use wrapper to get the node_name_map for auto_LiRPA key conversion
        temp_wrap = ShiIBPModelWrapper(model=temp_model, input_shape=[1, 28, 28], eps=EPSILON, num_epochs=160, device=device)
        sd = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True)
        
        mapped_sd = {}
        for k, v in sd.items():
            if k in temp_wrap.bounded_model.node_name_map:
                mapped_sd[temp_wrap.bounded_model.node_name_map[k]] = v
        
        # Load the properly mapped weights into a pure PyTorch model
        model = CNN7_Shi(in_shape=(1, 28, 28), n_classes=10).to(device)
        model.load_state_dict(mapped_sd, strict=True)
    else:
        print(f"Error: {MODEL_WEIGHTS_PATH} not found.")
        return

    # Now wrap the explicit ReLUs with MaskedReLUs for ablation dynamically for CNN7_Shi
    in_features = 0
    is_conv = True
    for idx, module in enumerate(model.layers):
        if isinstance(module, nn.Conv2d):
            in_features = module.out_channels
            is_conv = True
        elif isinstance(module, nn.Linear):
            in_features = module.out_features
            is_conv = False
        elif isinstance(module, nn.ReLU):
            model.layers[idx] = MaskedReLU(module, in_features, is_conv=is_conv).to(device)
            
    model.eval()

    # Wrap the model using ShiIBPModelWrapper to use its evaluate function
    wrapped_model = ShiIBPModelWrapper(model=model, input_shape=[1, 28, 28], eps=EPSILON, num_epochs=160, device=device)
    
    # Baseline
    print("Evaluating Baseline on Identification Split (1000 test samples)...")
    base_clean, base_cert = wrapped_model.evaluate(ident_loader)
    print(f"Baseline Clean Acc: {base_clean:.4f}, Certified Acc: {base_cert:.4f}")
    
    orig_clean = base_clean
    orig_cert = base_cert

    # Collect all layers and neurons
    layers_ordered = []
    for idx, module in enumerate(model.layers):
        if isinstance(module, MaskedReLU):
            layers_ordered.append(f'layers.{idx}')
            
    # Iterate in reverse order
    layers_ordered.reverse()
    
    ablated_neurons = []
    
    channel_accuracies_clean = []
    channel_accuracies_cert = []
    
    channel_base_clean = []
    channel_base_cert = []
    
    layer_boundaries = []
    layer_labels = []
    
    current_x = 0
    
    for l_idx, layer_name in enumerate(layers_ordered):
        print(f"Processing layer {layer_name} ({l_idx + 1}/{len(layers_ordered)})")
        
        idx = int(layer_name.split('.')[1])
        mask = model.layers[idx].mask
        num_neurons = mask.shape[1]
            
        layer_start_x = current_x
            
        for n_idx in range(num_neurons):
            channel_base_clean.append(base_clean)
            channel_base_cert.append(base_cert)
            
            mask[0, n_idx] = 0.0
            
            new_clean, new_cert = wrapped_model.evaluate(ident_loader)
            channel_accuracies_clean.append(new_clean)
            channel_accuracies_cert.append(new_cert)
            
            if (new_clean > base_clean and new_cert > base_cert) or (new_clean >= base_clean and new_cert > base_cert) or (new_clean > base_clean and new_cert >= base_cert):
                ablated_neurons.append([layer_name, n_idx])
                print(f"[{l_idx+1}/{len(layers_ordered)}] Ablated {layer_name}:{n_idx} | Clean: {new_clean:.4f} (base: {base_clean:.4f}), Cert: {new_cert:.4f} (base: {base_cert:.4f}) | KEPT (Definetely Ablated)")
                base_clean = new_clean
                base_cert = new_cert
            else:
                mask[0, n_idx] = 1.0
                print(f"[{l_idx+1}/{len(layers_ordered)}] Ablated {layer_name}:{n_idx} | Clean: {new_clean:.4f} (base: {base_clean:.4f}), Cert: {new_cert:.4f} (base: {base_cert:.4f}) | REVERTED")
                
            current_x += 1
            
        layer_boundaries.append((layer_start_x, current_x))
        layer_labels.append(str(len(layers_ordered) - l_idx))

    out_file = os.path.join(PROJECT_ROOT, 'jsons', 'cnn7', 'mnist', 'cnn7_mnist_mtl_ibp_harmful_channels.json')
    with open(out_file, 'w') as f:
        json.dump(ablated_neurons, f, indent=4)
        
    print(f"Finished. Identified {len(ablated_neurons)} neurons to definetely ablate. Saved to {out_file}")
    
    

if __name__ == "__main__":
    main()
