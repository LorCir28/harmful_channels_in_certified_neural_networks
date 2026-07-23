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


from ctrain.model_wrappers.sabr_model_wrapper import SABRModelWrapper
from ctrain.data_loaders import load_cifar10
from source.training.resnet18.cifar.resnet18_cifar_sabr_training import CIFAR_ResNet18

# Prevent re-initializing the weights of loaded model in the training loop
import ctrain.train.certified.sabr
ctrain.train.certified.sabr.ibp_init_shi = lambda *args, **kwargs: None
ctrain.train.certified.sabr.save_checkpoint = lambda *args, **kwargs: None

def main():
    # Hyperparameters
    BATCH_SIZE = 128
    EPSILON = 8 / 255
    NUM_EPOCHS = 10
    LR = 0.00003
    LAMBDA_REG = 0.001  # L2 regularization strength
    TOP_K = 0.15        # Fraction of channels to regularize (15%)
    SABR_SELECTION_RATIO_LAMBDA = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load CIFAR-10 dataset
    print("Loading CIFAR-10 data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader_full, test_loader_full = load_cifar10(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    # Use entire train set (50,000 samples) and entire test set (10,000 samples)
    finetune_loader = train_loader_full
    eval_loader = test_loader_full

    # Copy necessary custom attributes for the wrapper / bounds calculator
    for loader in [finetune_loader, eval_loader]:
        if hasattr(test_loader_full, 'normalised'):
            loader.normalised = test_loader_full.normalised
            loader.mean = getattr(test_loader_full, 'mean', None)
            loader.std = test_loader_full.std
            loader.min = test_loader_full.min
            loader.max = test_loader_full.max

    print(f"Using {len(finetune_loader.dataset)} samples for training.")
    print(f"Using {len(eval_loader.dataset)} samples for evaluation.")

    # 2. Parse attribution JSON and normalize scores (Interpretation B with Top-K selection)
    json_path = os.path.join(PROJECT_ROOT, 'jsons', 'resnet18', 'cifar', 'resnet18_cifar_sabr_scores.json')
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

    # Sort descending by score to identify channels by fragility score
    pos_attributions.sort(key=lambda x: x['score'], reverse=True)

    # Select top-K channels
    num_to_keep = int(len(pos_attributions) * TOP_K)
    kept_attributions = pos_attributions[:num_to_keep]
    print(f"Selecting TOP {TOP_K*100:.1f}% channels for L2 regularization ({num_to_keep} out of {len(pos_attributions)} positive channels).")

    # Find the maximum positive score overall for global relative normalization
    max_pos_score = pos_attributions[0]['score'] if pos_attributions else 1.0
    if max_pos_score == 0.0:
        max_pos_score = 1.0

    # Map auto-LiRPA BoundRelu nodes to PyTorch names
    node_to_pytorch = {
        '/input-8': 'relu1',
        '/input-20': 'layer1.0.relu1',
        '/input-28': 'layer1.0.relu2',
        '/input-40': 'layer1.1.relu1',
        '/input-48': 'layer1.1.relu2',
        '/input-60': 'layer2.0.relu1',
        '/input-72': 'layer2.0.relu2',
        '/input-84': 'layer2.1.relu1',
        '/input-92': 'layer2.1.relu2',
        '/input-104': 'layer3.0.relu1',
        '/input-116': 'layer3.0.relu2',
        '/input-128': 'layer3.1.relu1',
        '/input-136': 'layer3.1.relu2',
        '/input-148': 'layer4.0.relu1',
        '/input-160': 'layer4.0.relu2',
        '/input-172': 'layer4.1.relu1',
        '/input-180': 'layer4.1.relu2',
    }

    # Pre-allocate score maps for Conv and BN layers (defaulting to 0)
    layer_channels = {
        'conv1': 64, 'bn1': 64,
        'layer1.0.conv1': 64, 'layer1.0.bn1': 64, 'layer1.0.conv2': 64, 'layer1.0.bn2': 64,
        'layer1.1.conv1': 64, 'layer1.1.bn1': 64, 'layer1.1.conv2': 64, 'layer1.1.bn2': 64,
        'layer2.0.conv1': 128, 'layer2.0.bn1': 128, 'layer2.0.conv2': 128, 'layer2.0.bn2': 128,
        'layer2.1.conv1': 128, 'layer2.1.bn1': 128, 'layer2.1.conv2': 128, 'layer2.1.bn2': 128,
        'layer3.0.conv1': 256, 'layer3.0.bn1': 256, 'layer3.0.conv2': 256, 'layer3.0.bn2': 256,
        'layer3.1.conv1': 256, 'layer3.1.bn1': 256, 'layer3.1.conv2': 256, 'layer3.1.bn2': 256,
        'layer4.0.conv1': 512, 'layer4.0.bn1': 512, 'layer4.0.conv2': 512, 'layer4.0.bn2': 512,
        'layer4.1.conv1': 512, 'layer4.1.bn1': 512, 'layer4.1.conv2': 512, 'layer4.1.bn2': 512,
    }
    
    scores_map = {name: torch.zeros(ch) for name, ch in layer_channels.items()}

    # Populate only for the selected top_k channels
    for item in kept_attributions:
        node_name = item['layer']
        if node_name in node_to_pytorch:
            relu_name = node_to_pytorch[node_name]
            conv_name = relu_name.replace('relu', 'conv')
            bn_name = relu_name.replace('relu', 'bn')
            
            ch_idx = item['channel']
            raw_score = item['score']
            norm_score = raw_score / max_pos_score
            
            if conv_name in scores_map and ch_idx < len(scores_map[conv_name]):
                scores_map[conv_name][ch_idx] = norm_score
            if bn_name in scores_map and ch_idx < len(scores_map[bn_name]):
                scores_map[bn_name][ch_idx] = norm_score

    # Move scores map to device for training
    device_scores_map = {name: s.to(device) for name, s in scores_map.items()}

    # 3. Define the custom L2 regularization function
    def score_weighted_l2_reg(model=None, device=None, **kwargs):
        penalty = 0.0
        for name, module in model.named_modules():
            if name in device_scores_map:
                s = device_scores_map[name]
                if isinstance(module, nn.Conv2d):
                    w_squared = (module.weight ** 2).sum(dim=(1, 2, 3))
                    penalty += (s * w_squared).sum()
                elif isinstance(module, nn.BatchNorm2d):
                    w_squared = module.weight ** 2
                    if module.bias is not None:
                        w_squared += module.bias ** 2
                    penalty += (s * w_squared).sum()
        return penalty

    # Override the regularizer in the sabr module
    ctrain.train.certified.sabr.get_l1_reg = score_weighted_l2_reg

    # 4. Construct ResNet18 model & wrap it
    model = CIFAR_ResNet18(num_classes=10).to(device)
    
    in_shape = [3, 32, 32]
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'cifar')
    os.makedirs(out_dir, exist_ok=True)

    wrapped_model = SABRModelWrapper(
        model=model, 
        input_shape=in_shape, 
        eps=EPSILON, 
        num_epochs=NUM_EPOCHS,
        lr=LR,
        warm_up_epochs=0,
        ramp_up_epochs=0,
        lr_decay_factor=0.5,
        lr_decay_milestones=(5, 8),
        checkpoint_save_path=out_dir,
        device=device,
        sabr_subselection_ratio=SABR_SELECTION_RATIO_LAMBDA,
        l1_reg_weight=LAMBDA_REG
    )

    # 5. Load pre-trained weights
    pretrained_path = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'cifar', 'resnet18_cifar_sabr_8.pt')
        
    print(f"Loading pretrained weights from {pretrained_path}...")
    wrapped_model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

    # Make sure all parameters are trainable (no freezing)
    for param in model.parameters():
        param.requires_grad = True

    # 6. Evaluate the original model before finetuning
    print("\n--- Evaluating Pre-trained Original Model on entire test set ---")
    wrapped_model.eval()
    orig_res = wrapped_model.evaluate(eval_loader, test_samples=10000)
    print(f"Original Model - Clean Acc: {orig_res[0]:.4f}, Certified Acc: {orig_res[1]:.4f}")

    # 7. Start fine-tuning
    print(f"\n--- Starting Fine-Tuning with score-weighted L2 regularization ---")
    for epoch in range(0, NUM_EPOCHS, 5):
        end_epoch = min(epoch + 5, NUM_EPOCHS)
        wrapped_model.train_model(finetune_loader, start_epoch=epoch, end_epoch=end_epoch)
        
        print(f"--- Evaluating after Epoch {end_epoch} ---")
        wrapped_model.eval()
        res = wrapped_model.evaluate(eval_loader, test_samples=10000)
        print(f"Clean Acc: {res[0]:.4f}, Certified Acc: {res[1]:.4f}")

    # 8. Final evaluation
    print("\n--- Final Evaluation of Finetuned Model on entire test set ---")
    wrapped_model.eval()
    final_res = wrapped_model.evaluate(eval_loader, test_samples=10000)
    print(f"Finetuned Model - Clean Acc: {final_res[0]:.4f}, Certified Acc: {final_res[1]:.4f}")
    
    # Save the final fine-tuned model
    out_path = os.path.join(out_dir, f'resnet18_cifar_sabr_l2.pt')
    torch.save(wrapped_model.state_dict(), out_path)
    print(f"Finetuned model saved to {out_path}")

if __name__ == "__main__":
    main()
