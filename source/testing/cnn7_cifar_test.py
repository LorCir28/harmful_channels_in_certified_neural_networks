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

from ctrain.model_wrappers.crown_ibp_model_wrapper import CrownIBPModelWrapper
from ctrain.data_loaders import load_cifar10
from ctrain.model_definitions.models_shi import CNN7_Shi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Options & Configuration
# ==========================================
TEST_SAMPLES = 1000
EPSILON = 8 / 255
BATCH_SIZE = 128
NUM_RUNS = 20

MODELS = {
    'ibp': {
        'pt': 'cnn7_cifar_ibp_8.pt',
        'json': 'cnn7_cifar_ibp_harmful_channels.json',
        'display_name': 'IBP'
    },
    'crown-ibp': {
        'pt': 'cnn7_cifar_crown_ibp_8.pt',
        'json': 'cnn7_cifar_crown_ibp_harmful_channels.json',
        'display_name': 'CROWN-IBP'
    },
    'sabr': {
        'pt': 'cnn7_cifar_sabr_8.pt',
        'json': 'cnn7_cifar_sabr_harmful_channels.json',
        'display_name': 'SABR'
    },
    'mtl-ibp': {
        'pt': 'cnn7_cifar_mtl_ibp_8.pt',
        'json': 'cnn7_cifar_mtl_ibp_harmful_channels.json',
        'display_name': 'MTL-IBP'
    }
}

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
        for layer in model.layers:
            if isinstance(layer, MaskedReLU):
                layer.mask.fill_(1.0)

def main():
    print(f"Using device: {device}")

    # 1. Load data
    print("Loading CIFAR-10 data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    _, test_loader_full = load_cifar10(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

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
    in_shape = (3, 32, 32)

    for model_key, cfg in MODELS.items():
        print(f"\n==================================================")
        print(f" Processing model: {cfg['display_name']}")
        print(f"==================================================")
        
        model_path = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'cifar', cfg['pt'])
        json_path = os.path.join(PROJECT_ROOT, 'jsons', 'cnn7', 'cifar', cfg['json'])
        
        if not os.path.exists(model_path):
            print(f"Weights not found at {model_path}, skipping.")
            continue

        # 2. Load model weights with key mapping from auto_LiRPA to PyTorch
        temp_model = CNN7_Shi(in_shape=in_shape, n_classes=10).to(device)
        temp_wrap = CrownIBPModelWrapper(model=temp_model, input_shape=in_shape, eps=EPSILON, num_epochs=1, device=device)
        sd = torch.load(model_path, map_location=device, weights_only=True)
        
        mapped_sd = {}
        for k, v in sd.items():
            if k in temp_wrap.bounded_model.node_name_map:
                mapped_sd[temp_wrap.bounded_model.node_name_map[k]] = v
                
        model = CNN7_Shi(in_shape=in_shape, n_classes=10).to(device)
        model.load_state_dict(mapped_sd, strict=True)

        # 3. Wrap ReLUs dynamically with MaskedReLUs for ablation
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

        # 4. Wrap the model using CrownIBPModelWrapper
        wrapped_model = CrownIBPModelWrapper(
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
                    layer_idx = int(layer_name.split('.')[-1])
                    if isinstance(model.layers[layer_idx], MaskedReLU):
                        model.layers[layer_idx].mask[0, neuron_idx] = 0.0
                        json_ablated_set.add((layer_idx, neuron_idx))
                        json_ablated_by_layer.setdefault(layer_idx, []).append(neuron_idx)
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

        # Collect all possible (layer_idx, channel_idx) across all MaskedReLU layers
        all_possible_neurons = []
        neurons_by_layer = {}
        for layer_idx, layer in enumerate(model.layers):
            if isinstance(layer, MaskedReLU):
                num_channels = layer.mask.shape[1]
                neurons_by_layer[layer_idx] = list(range(num_channels))
                for neuron_idx in range(num_channels):
                    all_possible_neurons.append((layer_idx, neuron_idx))

        # 7. Evaluation (With Global Random Zero Ablation across runs)
        print(f"\n--- Evaluation (With Global Random Zero Ablation across {NUM_RUNS} runs) ---")
        global_candidate_neurons = [pair for pair in all_possible_neurons if pair not in json_ablated_set]
        
        global_clean_accs = []
        global_cert_accs = []
        
        for run_idx in range(NUM_RUNS):
            reset_masks(model)
            selected = random.sample(global_candidate_neurons, min(ablated_count, len(global_candidate_neurons)))
            with torch.no_grad():
                for l_idx, ch_idx in selected:
                    model.layers[l_idx].mask[0, ch_idx] = 0.0
            
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
                for l_idx, target_channels in json_ablated_by_layer.items():
                    num_to_ablate = len(target_channels)
                    layer_candidates = [ch for ch in neurons_by_layer[l_idx] if ch not in target_channels]
                    selected_channels = random.sample(layer_candidates, min(num_to_ablate, len(layer_candidates)))
                    for ch_idx in selected_channels:
                        model.layers[l_idx].mask[0, ch_idx] = 0.0
                    
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


