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
from ctrain.data_loaders import load_cifar10

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

# Define Custom ResNet18 to ensure explicit ReLUs for masking
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

class CIFAR_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR_ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

def main():
    EPSILON = 8 / 255
    MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'cifar', 'resnet18_cifar_mtl_ibp_8.pt')
    # REMEMBER TO MODIFY THE OUTPUTTED JSON FILE NAME

    print("Loading CIFAR-10 data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    _, test_loader_full = load_cifar10(batch_size=128, val_split=False, data_root=data_root)

    # Use the 1,000 images from the test set (indices 0 to 1000) for identification
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
    temp_model = CIFAR_ResNet18(num_classes=10).to(device)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        # Use wrapper to get the node_name_map for auto_LiRPA key conversion
        temp_wrap = ShiIBPModelWrapper(model=temp_model, input_shape=[3, 32, 32], eps=EPSILON, num_epochs=160, device=device)
        sd = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
        
        mapped_sd = {}
        for k, v in sd.items():
            if k in temp_wrap.bounded_model.node_name_map:
                mapped_sd[temp_wrap.bounded_model.node_name_map[k]] = v
        
        # Load the properly mapped weights into a pure PyTorch model
        model = CIFAR_ResNet18(num_classes=10).to(device)
        model.load_state_dict(mapped_sd, strict=True)
    else:
        print(f"Error: {MODEL_WEIGHTS_PATH} not found.")
        return

    # Now wrap the explicit ReLUs with MaskedReLUs for ablation
    model.relu1 = MaskedReLU(model.relu1, 64, is_conv=True).to(device)
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for b_idx, block in enumerate(layer):
            block.relu1 = MaskedReLU(block.relu1, block.bn1.num_features, is_conv=True).to(device)
            block.relu2 = MaskedReLU(block.relu2, block.bn2.num_features, is_conv=True).to(device)
    
    # We no longer need to copy from temp_model, model already has weights!
    model.eval()

    # Wrap the model using ShiIBPModelWrapper to use its evaluate function
    wrapped_model = ShiIBPModelWrapper(model=model, input_shape=[3, 32, 32], eps=EPSILON, num_epochs=160, device=device)
    
    # Baseline
    print("Evaluating Baseline on Identification Split (1000 test samples)...")
    base_clean, base_cert = wrapped_model.evaluate(ident_loader)
    print(f"Baseline Clean Acc: {base_clean:.4f}, Certified Acc: {base_cert:.4f}")
    
    orig_clean = base_clean
    orig_cert = base_cert

    # Collect all layers and neurons
    # Create an ordered list of layers
    layers_ordered = ['relu1']
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for b_idx, _ in enumerate(layer):
            layers_ordered.append(f'{layer_name}.{b_idx}.relu1')
            layers_ordered.append(f'{layer_name}.{b_idx}.relu2')
            
    # Iterate in reverse order
    layers_ordered.reverse()
    
    ablated_neurons = []
    
    channel_accuracies_clean = []
    channel_accuracies_cert = []
    
    # We will record the best (baseline) clean/cert accuracy before ablating each channel to compute the weight
    channel_base_clean = []
    channel_base_cert = []
    
    layer_boundaries = []
    layer_labels = []
    
    current_x = 0
    
    for l_idx, layer_name in enumerate(layers_ordered):
        print(f"Processing layer {layer_name} ({l_idx + 1}/{len(layers_ordered)})")
        
        # Get number of neurons in this layer
        if '.' in layer_name:
            parts = layer_name.split('.')
            block_obj = getattr(model, parts[0])[int(parts[1])]
            relu_obj = getattr(block_obj, parts[2])
            mask = relu_obj.mask
            num_neurons = mask.shape[1]
        else:
            mask = getattr(model, layer_name).mask
            num_neurons = mask.shape[1]
            
        layer_start_x = current_x
            
        for n_idx in range(num_neurons):
            # Record current baseline to compute weight later
            channel_base_clean.append(base_clean)
            channel_base_cert.append(base_cert)
            
            # Zero ablate
            mask[0, n_idx] = 0.0
            
            new_clean, new_cert = wrapped_model.evaluate(ident_loader)
            channel_accuracies_clean.append(new_clean)
            channel_accuracies_cert.append(new_cert)
            
            if (new_clean > base_clean and new_cert > base_cert) or (new_clean >= base_clean and new_cert > base_cert) or (new_clean > base_clean and new_cert >= base_cert):
                # Definitely zero ablate
                ablated_neurons.append([layer_name, n_idx])
                print(f"[{l_idx+1}/{len(layers_ordered)}] Ablated {layer_name}:{n_idx} | Clean: {new_clean:.4f} (base: {base_clean:.4f}), Cert: {new_cert:.4f} (base: {base_cert:.4f}) | KEPT (Definetely Ablated)")
                # Update baseline
                base_clean = new_clean
                base_cert = new_cert
            else:
                # Revert
                mask[0, n_idx] = 1.0
                print(f"[{l_idx+1}/{len(layers_ordered)}] Ablated {layer_name}:{n_idx} | Clean: {new_clean:.4f} (base: {base_clean:.4f}), Cert: {new_cert:.4f} (base: {base_cert:.4f}) | REVERTED")
                
            current_x += 1
            
        layer_boundaries.append((layer_start_x, current_x))
        # Label with the architectural layer depth (17 down to 1)
        layer_labels.append(str(len(layers_ordered) - l_idx))

    # Save to JSON
    out_file = os.path.join(PROJECT_ROOT, 'jsons', 'resnet18', 'cifar', 'resnet18_cifar_mtl_ibp_harmful_channels.json')
    with open(out_file, 'w') as f:
        json.dump(ablated_neurons, f, indent=4)
        
    print(f"Finished. Identified {len(ablated_neurons)} neurons to definetely ablate. Saved to {out_file}")
    


if __name__ == "__main__":
    main()