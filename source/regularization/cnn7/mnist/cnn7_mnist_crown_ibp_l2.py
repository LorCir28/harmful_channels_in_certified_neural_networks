import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure ctrain is in the path
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.model_wrappers.crown_ibp_model_wrapper import CrownIBPModelWrapper
from ctrain.data_loaders import load_mnist
from ctrain.model_definitions.models_shi import CNN7_Shi

# Prevent re-initializing the weights of loaded model in the training loop
import ctrain.train.certified.crown_ibp
ctrain.train.certified.crown_ibp.ibp_init_shi = lambda *args, **kwargs: None
ctrain.train.certified.crown_ibp.save_checkpoint = lambda *args, **kwargs: None

def get_device_scores_map(wrapped_model, pos_attributions, top_k, max_pos_score, device):
    # Find the ReLU nodes in the bounded module
    relu_node_names = []
    for node in wrapped_model.bounded_model.nodes():
        if 'BoundRelu' in str(type(node)):
            relu_node_names.append(node.name)
            
    # We expect 6 ReLU layers in CNN7
    if len(relu_node_names) > 6:
        relu_node_names = relu_node_names[:6]
        
    pytorch_relus = ['layers.2', 'layers.5', 'layers.8', 'layers.11', 'layers.14', 'layers.18']
    node_to_pytorch = {}
    for r_node, r_py in zip(relu_node_names, pytorch_relus):
        node_to_pytorch[r_node] = r_py

    # Pre-allocate score maps for Conv, Linear, and BN layers (defaulting to 0)
    layer_channels = {}
    for name, module in wrapped_model.original_model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_channels[name] = module.out_channels
        elif isinstance(module, nn.Linear):
            layer_channels[name] = module.out_features
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            layer_channels[name] = module.num_features

    scores_map = {name: torch.zeros(ch) for name, ch in layer_channels.items()}

    num_to_keep = int(len(pos_attributions) * top_k)
    kept_attributions = pos_attributions[:num_to_keep]

    # Populate only for the selected top_k channels
    for item in kept_attributions:
        node_name = item['layer']
        if node_name in node_to_pytorch:
            relu_name = node_to_pytorch[node_name]
            parts = relu_name.split('.')
            idx = int(parts[1])
            conv_linear_name = f"layers.{idx-2}"
            bn_name = f"layers.{idx-1}"
            
            ch_idx = item['channel']
            raw_score = item['score']
            norm_score = raw_score / max_pos_score
            
            if conv_linear_name in scores_map and ch_idx < len(scores_map[conv_linear_name]):
                scores_map[conv_linear_name][ch_idx] = norm_score
            if bn_name in scores_map and ch_idx < len(scores_map[bn_name]):
                scores_map[bn_name][ch_idx] = norm_score

    # Move scores map to device for training
    device_scores_map = {name: s.to(device) for name, s in scores_map.items()}
    return device_scores_map

def main():
    # Hyperparameters & Macros from training script
    BATCH_SIZE = 256
    EPSILON = 0.3
    NUM_EPOCHS = 10
    
    GRADIENT_CLIP = 10
    SHI_REG_WEIGHT = 0.0
    START_BETA = 1.0
    END_BETA = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load MNIST dataset
    print("Loading MNIST data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader_full, test_loader_full = load_mnist(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    finetune_loader = train_loader_full
    eval_loader = test_loader_full

    for loader in [finetune_loader, eval_loader]:
        if hasattr(test_loader_full, 'normalised'):
            loader.normalised = test_loader_full.normalised
            loader.mean = getattr(test_loader_full, 'mean', None)
            loader.std = test_loader_full.std
            loader.min = test_loader_full.min
            loader.max = test_loader_full.max

    print(f"Using {len(finetune_loader.dataset)} samples for training.")
    print(f"Using {len(eval_loader.dataset)} samples for evaluation.")

    # 2. Parse attribution JSON
    json_path = os.path.join(PROJECT_ROOT, 'jsons', 'cnn7', 'mnist', 'cnn7_mnist_crown_ibp_scores.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            attributions = json.load(f)
    else:
        print(f"Error: {json_path} not found.")
        return

    # Extract all positive scores
    pos_attributions = []
    for item in attributions:
        score = item['score']
        if score > 0:
            pos_attributions.append(item)

    pos_attributions.sort(key=lambda x: x['score'], reverse=True)
    max_pos_score = pos_attributions[0]['score'] if pos_attributions else 1.0
    if max_pos_score == 0.0:
        max_pos_score = 1.0

    in_shape = [1, 28, 28]
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'mnist')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cnn7_mnist_crown_ibp_l2.pt')

    pretrained_path = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'mnist', 'cnn7_mnist_crown_ibp_3.pt')


    # 3. Evaluate baseline accuracies dynamically
    print("Evaluating baseline CNN7 model on entire test set...")
    temp_model = CNN7_Shi(in_shape=in_shape, width=64, linear_size=512, n_classes=10).to(device)
    base_wrapper = CrownIBPModelWrapper(
        model=temp_model, 
        input_shape=in_shape, 
        eps=EPSILON, 
        num_epochs=1,
        checkpoint_save_path=None,
        device=device,
        gradient_clip=GRADIENT_CLIP,
        l1_reg_weight=0.0,
        shi_reg_weight=SHI_REG_WEIGHT,
        start_beta=START_BETA,
        end_beta=END_BETA
    )
    base_wrapper.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    base_wrapper.eval()
    base_res = base_wrapper.evaluate(eval_loader, test_samples=10000)
    baseline_clean = base_res[0] * 100.0
    baseline_cert = base_res[1] * 100.0
    print(f"Baseline - Clean Acc: {baseline_clean:.4f}%, Certified Acc: {baseline_cert:.4f}%")

    def run_fine_tune(top_k, lambda_reg, lr, num_epochs):
        model = CNN7_Shi(in_shape=in_shape, width=64, linear_size=512, n_classes=10).to(device)
        wrapped_model = CrownIBPModelWrapper(
            model=model, 
            input_shape=in_shape, 
            eps=EPSILON, 
            num_epochs=num_epochs,
            lr=lr,
            warm_up_epochs=0,
            ramp_up_epochs=0,
            lr_decay_factor=0.5,
            lr_decay_milestones=(5, 8),
            checkpoint_save_path=out_dir,
            device=device,
            l1_reg_weight=lambda_reg,
            gradient_clip=GRADIENT_CLIP,
            shi_reg_weight=SHI_REG_WEIGHT,
            start_beta=START_BETA,
            end_beta=END_BETA
        )
        wrapped_model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
        for param in model.parameters():
            param.requires_grad = True

        device_scores_map = get_device_scores_map(wrapped_model, pos_attributions, top_k, max_pos_score, device)

        def score_weighted_l2_reg(model=None, device=None, **kwargs):
            penalty = 0.0
            for name, module in model.named_modules():
                if name in device_scores_map:
                    s = device_scores_map[name]
                    if isinstance(module, nn.Conv2d):
                        w_squared = (module.weight ** 2).sum(dim=(1, 2, 3))
                        penalty += (s * w_squared).sum()
                    elif isinstance(module, nn.Linear):
                        w_squared = (module.weight ** 2).sum(dim=1)
                        penalty += (s * w_squared).sum()
                    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        w_squared = module.weight ** 2
                        if module.bias is not None:
                            w_squared += module.bias ** 2
                        penalty += (s * w_squared).sum()
            return penalty

        # Override regularizer
        ctrain.train.certified.crown_ibp.get_l1_reg = score_weighted_l2_reg

        wrapped_model.train_model(finetune_loader, start_epoch=0, end_epoch=num_epochs)
        wrapped_model.eval()
        res = wrapped_model.evaluate(eval_loader, test_samples=10000)
        clean = res[0] * 100.0
        cert = res[1] * 100.0
        return wrapped_model, clean, cert


    best_top_k = 0.15
    best_lambda_reg = 0.0002
    best_lr = 0.00003


    best_model, final_clean, final_cert = run_fine_tune(best_top_k, best_lambda_reg, best_lr, num_epochs=NUM_EPOCHS)
    
    print(f"\n--- Final Results ---")
    print(f"Baseline - Clean Acc: {baseline_clean:.4f}%, Certified Acc: {baseline_cert:.4f}%")
    print(f"Finetuned - Clean Acc: {final_clean:.4f}%, Certified Acc: {final_cert:.4f}%")
    

    # Save the final fine-tuned model
    torch.save(best_model.state_dict(), out_path)
    print(f"Finetuned model saved to {out_path}")

if __name__ == "__main__":
    main()
