import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ctrain.model_wrappers import ShiIBPModelWrapper
from ctrain.data_loaders import load_mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Options & Configuration
# ==========================================
TEST_SAMPLES = 1000
EPSILON = 0.3
BATCH_SIZE = 128
NUM_RUNS = 20

MODELS = {
    'ibp': {
        'pt': 'resnet18_mnist_ibp_3.pt',
        'json': 'resnet18_mnist_ibp_harmful_channels.json',
        'display_name': 'IBP'
    },
    'crown-ibp': {
        'pt': 'resnet18_mnist_crown_ibp_3.pt',
        'json': 'resnet18_mnist_crown_ibp_harmful_channels.json',
        'display_name': 'CROWN-IBP'
    },
    'sabr': {
        'pt': 'resnet18_mnist_sabr_3.pt',
        'json': 'resnet18_mnist_sabr_harmful_channels.json',
        'display_name': 'SABR'
    },
    'mtl-ibp': {
        'pt': 'resnet18_mnist_mtl_ibp_3.pt',
        'json': 'resnet18_mnist_mtl_ibp_harmful_channels.json',
        'display_name': 'MTL-IBP'
    }
}

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

class MNIST_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(MNIST_ResNet18, self).__init__()
        self.in_planes = 64

        # 1 input channel for MNIST
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
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

class MaskedReLU(nn.Module):
    def __init__(self, relu, num_neurons, is_conv):
        super().__init__()
        self.relu = relu
        shape = (1, num_neurons, 1, 1) if is_conv else (1, num_neurons)
        self.mask = nn.Parameter(torch.ones(shape, device=device))
        self.mask.requires_grad = False
        
    def forward(self, x):
        return self.relu(x) * self.mask

def reset_masks(model):
    with torch.no_grad():
        model.relu1.mask.fill_(1.0)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            for block in layer:
                block.relu1.mask.fill_(1.0)
                block.relu2.mask.fill_(1.0)

def get_relu_by_name(model, layer_name):
    if '.' in layer_name:
        parts = layer_name.split('.')
        block_obj = getattr(model, parts[0])[int(parts[1])]
        return getattr(block_obj, parts[2])
    else:
        return getattr(model, layer_name)

def main():
    print(f"Using device: {device}")

    # 1. Load data
    print("Loading MNIST data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    _, test_loader_full = load_mnist(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    # Explicitly use the first 1000 images (indices 0 to 1000) for deterministic evaluation
    from torch.utils.data import Subset, DataLoader
    test_ds = test_loader_full.dataset
    ident_ds = Subset(test_ds, range(0, TEST_SAMPLES))
    test_loader = DataLoader(ident_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Copy metadata attributes required by ctrain wrapper evaluation
    if hasattr(test_loader_full, 'normalised'):
        test_loader.normalised = test_loader_full.normalised
        test_loader.mean = getattr(test_loader_full, 'mean', None)
        test_loader.std = test_loader_full.std
        test_loader.min = test_loader_full.min
        test_loader.max = test_loader_full.max

    print(f"Using {len(test_loader.dataset)} samples for evaluation.")

    results = {}
    in_shape = (1, 28, 28)

    for model_key, cfg in MODELS.items():
        print(f"\n==================================================")
        print(f" Processing model: {cfg['display_name']}")
        print(f"==================================================")
        
        model_path = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'mnist', cfg['pt'])
        json_path = os.path.join(PROJECT_ROOT, 'jsons', 'resnet18', 'mnist', cfg['json'])
        
        if not os.path.exists(model_path):
            print(f"Weights not found at {model_path}, skipping.")
            continue

        # 2. Load model weights with key mapping from auto_LiRPA to PyTorch
        temp_model = MNIST_ResNet18(num_classes=10).to(device)
        temp_wrap = ShiIBPModelWrapper(model=temp_model, input_shape=in_shape, eps=EPSILON, num_epochs=1, device=device)
        sd = torch.load(model_path, map_location=device, weights_only=True)
        
        mapped_sd = {}
        for k, v in sd.items():
            if k in temp_wrap.bounded_model.node_name_map:
                mapped_sd[temp_wrap.bounded_model.node_name_map[k]] = v
                
        model = MNIST_ResNet18(num_classes=10).to(device)
        model.load_state_dict(mapped_sd, strict=True)

        # 3. Wrap ReLUs dynamically with MaskedReLUs for ablation
        model.relu1 = MaskedReLU(model.relu1, 64, is_conv=True).to(device)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            for b_idx, block in enumerate(layer):
                block.relu1 = MaskedReLU(block.relu1, block.bn1.num_features, is_conv=True).to(device)
                block.relu2 = MaskedReLU(block.relu2, block.bn2.num_features, is_conv=True).to(device)
                
        model.eval()

        # 4. Wrap the model using ShiIBPModelWrapper
        wrapped_model = ShiIBPModelWrapper(
            model=model, 
            input_shape=in_shape, 
            eps=EPSILON, 
            num_epochs=1, 
            device=device
        )
        wrapped_model.eval()

        # 5. Baseline Evaluation (No Ablation)
        reset_masks(model)
        print("\n--- Evaluation (No Ablation) ---")
        base_clean, base_cert = wrapped_model.evaluate(test_loader, test_samples=TEST_SAMPLES)
        base_clean *= 100.0
        base_cert *= 100.0
        print(f"Standard Accuracy: {base_clean:.2f}%")
        print(f"Certified Accuracy (IBP): {base_cert:.2f}%")

        # 6. Evaluation (With Zero Ablation from JSON)
        ab_clean, ab_cert = None, None
        ablated_count = 0
        json_ablated_set = set()
        json_ablated_by_layer = {}
        
        if os.path.exists(json_path):
            print(f"Loading ablation channels from {json_path}...")
            with open(json_path, 'r') as f:
                ablated_list = json.load(f)
            
            reset_masks(model)
            with torch.no_grad():
                for item in ablated_list:
                    layer_name, neuron_idx = item
                    relu_obj = get_relu_by_name(model, layer_name)
                    if isinstance(relu_obj, MaskedReLU):
                        relu_obj.mask[0, neuron_idx] = 0.0
                        json_ablated_set.add((layer_name, neuron_idx))
                        json_ablated_by_layer.setdefault(layer_name, []).append(neuron_idx)
                        ablated_count += 1
            
            print(f"Ablating {ablated_count} channels/neurons reported in JSON.")
            print(f"--- Evaluation (With JSON Zero Ablation) ---")
            ab_clean, ab_cert = wrapped_model.evaluate(test_loader, test_samples=TEST_SAMPLES)
            ab_clean *= 100.0
            ab_cert *= 100.0
            print(f"JSON Ablated Standard Accuracy: {ab_clean:.2f}%")
            print(f"JSON Ablated Certified Accuracy (IBP): {ab_cert:.2f}%")
        else:
            print(f"Error: JSON file not found at {json_path}")

        # Collect all possible (layer_name, channel_idx) across all MaskedReLU layers
        all_possible_neurons = []
        neurons_by_layer = {}

        # relu1
        neurons_by_layer['relu1'] = list(range(model.relu1.mask.shape[1]))
        for neuron_idx in range(model.relu1.mask.shape[1]):
            all_possible_neurons.append(('relu1', neuron_idx))

        # layer1 to layer4
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            for b_idx, block in enumerate(layer):
                r1_name = f'{layer_name}.{b_idx}.relu1'
                r2_name = f'{layer_name}.{b_idx}.relu2'
                
                neurons_by_layer[r1_name] = list(range(block.relu1.mask.shape[1]))
                for neuron_idx in range(block.relu1.mask.shape[1]):
                    all_possible_neurons.append((r1_name, neuron_idx))
                    
                neurons_by_layer[r2_name] = list(range(block.relu2.mask.shape[1]))
                for neuron_idx in range(block.relu2.mask.shape[1]):
                    all_possible_neurons.append((r2_name, neuron_idx))

        # 7. Evaluation (With Global Random Zero Ablation across runs)
        print(f"\n--- Evaluation (With Global Random Zero Ablation across {NUM_RUNS} runs) ---")
        global_candidate_neurons = [pair for pair in all_possible_neurons if pair not in json_ablated_set]
        
        global_clean_accs = []
        global_cert_accs = []
        
        for run_idx in range(NUM_RUNS):
            reset_masks(model)
            selected = random.sample(global_candidate_neurons, min(ablated_count, len(global_candidate_neurons)))
            with torch.no_grad():
                for l_name, ch_idx in selected:
                    relu_obj = get_relu_by_name(model, l_name)
                    relu_obj.mask[0, ch_idx] = 0.0
            
            clean_acc, cert_acc = wrapped_model.evaluate(test_loader, test_samples=TEST_SAMPLES)
            global_clean_accs.append(clean_acc * 100.0)
            global_cert_accs.append(cert_acc * 100.0)
            print(f"Global Run {run_idx + 1}/{NUM_RUNS}: Standard Acc: {clean_acc * 100.0:.2f}%, Certified Acc: {cert_acc * 100.0:.2f}%")

        # 8. Evaluation (With Layer-Matched Random Zero Ablation across runs)
        print(f"\n--- Evaluation (With Layer-Matched Random Zero Ablation across {NUM_RUNS} runs) ---")
        layer_matched_clean_accs = []
        layer_matched_cert_accs = []
        
        for run_idx in range(NUM_RUNS):
            reset_masks(model)
            with torch.no_grad():
                for l_name, target_channels in json_ablated_by_layer.items():
                    num_to_ablate = len(target_channels)
                    layer_candidates = [ch for ch in neurons_by_layer[l_name] if ch not in target_channels]
                    selected_channels = random.sample(layer_candidates, min(num_to_ablate, len(layer_candidates)))
                    relu_obj = get_relu_by_name(model, l_name)
                    for ch_idx in selected_channels:
                        relu_obj.mask[0, ch_idx] = 0.0
                    
            clean_acc, cert_acc = wrapped_model.evaluate(test_loader, test_samples=TEST_SAMPLES)
            layer_matched_clean_accs.append(clean_acc * 100.0)
            layer_matched_cert_accs.append(cert_acc * 100.0)
            print(f"Layer-Matched Run {run_idx + 1}/{NUM_RUNS}: Standard Acc: {clean_acc * 100.0:.2f}%, Certified Acc: {cert_acc * 100.0:.2f}%")

        # Compute stats function
        def get_stats(arr):
            arr = np.array(arr)
            mean_val = np.mean(arr)
            median_val = np.median(arr)
            std_val = np.std(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)
            return {
                'mean': float(mean_val),
                'median': float(median_val),
                'pos_std': float(mean_val + std_val),
                'neg_std': float(mean_val - std_val),
                'max': float(max_val),
                'min': float(min_val)
            }

        results[model_key] = {
            'base_clean': base_clean,
            'base_cert': base_cert,
            'ab_clean': ab_clean,
            'ab_cert': ab_cert,
            'global_clean_stats': get_stats(global_clean_accs),
            'global_cert_stats': get_stats(global_cert_accs),
            'layer_matched_clean_stats': get_stats(layer_matched_clean_accs),
            'layer_matched_cert_stats': get_stats(layer_matched_cert_accs),
        }

        # Print statistics summary to console
        print(f"\n==================================================")
        print(f" Statistics Summary for {cfg['display_name']}")
        print(f"==================================================")
        print(f"{'Metric / Config':<30} | {'Mean':<8} | {'Median':<8} | {'Neg Std':<8} | {'Pos Std':<8} | {'Min':<8} | {'Max':<8}")
        print("-" * 90)
        for stat_name, key in [
            ('Global Random Certified', 'global_cert_stats'),
            ('Layer-Matched Certified', 'layer_matched_cert_stats'),
            ('Global Random Standard', 'global_clean_stats'),
            ('Layer-Matched Standard', 'layer_matched_clean_stats'),
        ]:
            st = results[model_key][key]
            print(f"{stat_name:<30} | {st['mean']:7.2f}% | {st['median']:7.2f}% | {st['neg_std']:7.2f}% | {st['pos_std']:7.2f}% | {st['min']:7.2f}% | {st['max']:7.2f}%")
        print("==================================================\n")

if __name__ == "__main__":
    main()