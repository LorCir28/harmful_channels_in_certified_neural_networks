import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add the directory containing ctrain to the python path
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.data_loaders import load_cifar10
from ctrain.model_definitions.models_shi import CNN7_Shi
from ctrain.model_wrappers import ShiIBPModelWrapper
import auto_LiRPA

def compute_attribution_scores(model_path, loader, eps_std, epsilon, device):
    print(f"--- Loading and preparing CNN7 model ---")
    model = CNN7_Shi(in_shape=(3, 32, 32), width=64, linear_size=512, n_classes=10).to(device)
    wrapped_model = ShiIBPModelWrapper(model=model, input_shape=[3, 32, 32], eps=epsilon, num_epochs=160, device=device)
    
    sd = torch.load(model_path, map_location=device)
    wrapped_model.load_state_dict(sd, strict=False)
    
    bounded_model = wrapped_model.bounded_model
    bounded_model.eval()

    # Find the ReLU nodes in the bounded module
    relu_node_names = []
    for node in bounded_model.nodes():
        if isinstance(node, auto_LiRPA.operators.relu.BoundRelu):
            relu_node_names.append(node.name)
            
    # We expect 6 ReLU layers in CNN7
    if len(relu_node_names) > 6:
        relu_node_names = relu_node_names[:6]
        
    print(f"Found {len(relu_node_names)} ReLU nodes in BoundedModule: {relu_node_names}")

    # Dictionary to collect batch-level attribution scores per channel
    channel_attribution_batches = {name: [] for name in relu_node_names}

    print("Iterating over the first 1000 test samples...")
    processed_samples = 0
    
    for x, target in loader:
        x, target = x.to(device), target.to(device)
        x.requires_grad_()
        
        # Define the bounded input for auto_LiRPA
        ptc = auto_LiRPA.PerturbationLpNorm(norm=torch.inf, eps=eps_std)
        bounded_input = auto_LiRPA.BoundedTensor(x, ptc)
        
        # Forward pass: compute IBP bounds
        lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method='IBP', return_A=False)

        bounded_model.zero_grad()

        # Enable gradient tracking on intermediate ReLU bounds
        for name in relu_node_names:
            node = bounded_model[name]
            if hasattr(node, 'lower') and hasattr(node, 'upper'):
                node.lower.requires_grad_()
                node.upper.requires_grad_()
                node.lower.retain_grad()
                node.upper.retain_grad()

        # Compute robust margin: M = lb_correct - max_ub_others
        y_onehot = torch.nn.functional.one_hot(target, num_classes=lb.size(-1)).bool()
        lb_y = lb[y_onehot]
        ub_other = ub.masked_fill(y_onehot, -float('inf'))
        max_ub_other, _ = ub_other.max(dim=-1)
        margin = lb_y - max_ub_other

        # Backpropagate gradients of the sum of margins with respect to activation bounds
        margin.sum().backward()

        with torch.no_grad():
            for name in relu_node_names:
                node = bounded_model[name]
                if (hasattr(node, 'lower') and hasattr(node, 'upper') and 
                    node.lower.grad is not None and node.upper.grad is not None):
                    
                    # S = grad(M, a_LB) * (-a_LB) + grad(M, a_UB) * (-a_UB)
                    score_lower = node.lower.grad * (-node.lower)
                    score_upper = node.upper.grad * (-node.upper)
                    score_total = score_lower + score_upper
                    
                    # Average over spatial dimensions (H, W) to obtain channel scores
                    if score_total.dim() > 2:
                        score_total = score_total.mean(dim=list(range(2, score_total.dim())))
                        
                    channel_attribution_batches[name].append(score_total.cpu())

        processed_samples += x.size(0)
        # Clean up hooks/grad trackers to prevent memory leaks
        for name in relu_node_names:
            node = bounded_model[name]
            if hasattr(node, 'lower'): node.lower = None
            if hasattr(node, 'upper'): node.upper = None

    print(f"Finished processing {processed_samples} samples.")

    # Calculate final average scores per layer and per channel
    layer_scores = []
    channel_details = []

    for name in relu_node_names:
        # Concatenate across batches to get shape [1000, num_channels]
        all_channel_scores = torch.cat(channel_attribution_batches[name], dim=0)
        # Average over all 1000 samples
        mean_channel_scores = all_channel_scores.mean(dim=0)
        
        # The layer score is the mean of all channel scores in that layer
        avg_layer_score = mean_channel_scores.mean().item()
        layer_scores.append(avg_layer_score)
        
        for ch_idx, score in enumerate(mean_channel_scores):
            channel_details.append({
                "layer": name,
                "channel": ch_idx,
                "score": score.item()
            })

    return layer_scores, channel_details, relu_node_names

def main():
    BATCH_SIZE = 128
    EPSILON = 8 / 255
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10 data
    data_root = os.path.join(PROJECT_ROOT, 'data')
    _, test_loader_full = load_cifar10(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    # Use first 1000 samples of the test set
    test_ds = test_loader_full.dataset
    ident_ds = Subset(test_ds, range(0, 1000))
    ident_loader = DataLoader(ident_ds, batch_size=BATCH_SIZE, shuffle=False)

    if hasattr(test_loader_full, 'normalised'):
        ident_loader.normalised = test_loader_full.normalised
        ident_loader.mean = getattr(test_loader_full, 'mean', None)
        ident_loader.std = test_loader_full.std
        ident_loader.min = test_loader_full.min
        ident_loader.max = test_loader_full.max

    if hasattr(ident_loader, 'normalised') and ident_loader.normalised:
        eps_std = EPSILON / ident_loader.std
        eps_std = eps_std.view(1, 3, 1, 1).to(device)
    else:
        eps_std = torch.tensor(EPSILON).to(device)

    weights_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'cifar')
    jsons_dir = os.path.join(PROJECT_ROOT, 'jsons', 'cnn7', 'cifar')
    os.makedirs(jsons_dir, exist_ok=True)

    models_to_eval = [
        ("IBP", os.path.join(weights_dir, 'cnn7_cifar_ibp_8.pt')),
        ("CROWN-IBP", os.path.join(weights_dir, 'cnn7_cifar_crown_ibp_8.pt')),
        ("SABR", os.path.join(weights_dir, 'cnn7_cifar_sabr_8.pt')),
        ("MTL-IBP", os.path.join(weights_dir, 'cnn7_cifar_mtl_ibp_8.pt'))
    ]

    import numpy as np
    all_normalized_scores = {}
    relu_node_names_ref = None

    for model_name, model_path in models_to_eval:
        print(f"\n==========================================")
        print(f"Evaluating model: {model_name}")
        print(f"==========================================")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model weights not found at {model_path}")
            continue

        # Compute scores
        layer_scores, channel_details, relu_node_names = compute_attribution_scores(
            model_path, ident_loader, eps_std, EPSILON, device
        )
        
        if relu_node_names_ref is None:
            relu_node_names_ref = relu_node_names

        # Save details to JSON
        json_name = f'cnn7_cifar_{model_name.lower().replace("-", "_")}_scores.json'
        json_path = os.path.join(jsons_dir, json_name)
        with open(json_path, 'w') as f:
            json.dump(channel_details, f, indent=4)
        print(f"Saved channel-level scores to {json_path}")

        # Print summary per layer
        print(f"\n--- {model_name} Layer-wise Average Attribution Scores ---")
        for idx, (name, score) in enumerate(zip(relu_node_names, layer_scores)):
            print(f"Layer {idx+1:02d} ({name}): Average Score = {score:.6f}")

        # Normalize scores between 0 and 1 for the plot
        layer_scores_arr = np.array(layer_scores)
        min_score = layer_scores_arr.min()
        max_score = layer_scores_arr.max()
        score_range = max_score - min_score if max_score - min_score != 0 else 1.0
        normalized_scores = (layer_scores_arr - min_score) / score_range
        
        all_normalized_scores[model_name] = normalized_scores

    if not all_normalized_scores:
        print("No models were successfully evaluated.")
        return

if __name__ == '__main__':
    main()
