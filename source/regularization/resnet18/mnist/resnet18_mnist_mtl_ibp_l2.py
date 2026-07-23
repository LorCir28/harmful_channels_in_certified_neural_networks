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


from ctrain.model_wrappers.mtl_ibp_model_wrapper import MTLIBPModelWrapper
from ctrain.data_loaders import load_mnist
from source.training.resnet18.mnist.resnet18_mnist_mtl_ibp_training import MNIST_ResNet18

# Prevent re-initializing the weights of loaded model in the training loop
import ctrain.train.certified.mtl_ibp
ctrain.train.certified.mtl_ibp.ibp_init_shi = lambda *args, **kwargs: None
ctrain.train.certified.mtl_ibp.save_checkpoint = lambda *args, **kwargs: None

def get_device_scores_map(wrapped_model, pos_attributions, top_k, max_pos_score, device):
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

    num_to_keep = int(len(pos_attributions) * top_k)
    kept_attributions = pos_attributions[:num_to_keep]

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

    device_scores_map = {name: s.to(device) for name, s in scores_map.items()}
    return device_scores_map

def main():
    # Hyperparameters & Macros from training script
    BATCH_SIZE = 256
    EPSILON = 0.3
    NUM_EPOCHS = 10
    
    GRADIENT_CLIP = 10
    SHI_REG_WEIGHT = 0.5
    MTL_IBP_ALPHA = 0.08
    PGD_STEPS = 8
    PGD_ALPHA = 0.5
    PGD_RESTARTS = 1
    PGD_ALPHA_DECAY_FACTOR = 0.1
    PGD_DECAY_MILESTONES = (4, 7)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load MNIST dataset
    print("Loading MNIST data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader_full, test_loader_full = load_mnist(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    retrain_loader = train_loader_full
    eval_loader = test_loader_full

    for loader in [retrain_loader, eval_loader]:
        if hasattr(test_loader_full, 'normalised'):
            loader.normalised = test_loader_full.normalised
            loader.mean = getattr(test_loader_full, 'mean', None)
            loader.std = test_loader_full.std
            loader.min = test_loader_full.min
            loader.max = test_loader_full.max

    print(f"Using {len(retrain_loader.dataset)} samples for training.")
    print(f"Using {len(eval_loader.dataset)} samples for evaluation.")

    # 2. Parse attribution JSON
    json_path = os.path.join(PROJECT_ROOT, 'jsons', 'resnet18', 'mnist', 'resnet18_mnist_mtl_ibp_scores.json')
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
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'mnist')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'resnet18_mnist_mtl_ibp_l2.pt')

    pretrained_path = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'mnist', 'resnet18_mnist_mtl_ibp_3.pt')

    # 3. Evaluate baseline accuracies dynamically
    print("Evaluating baseline ResNet18 model on entire test set...")
    temp_model = MNIST_ResNet18(num_classes=10).to(device)
    base_wrapper = MTLIBPModelWrapper(
        model=temp_model, 
        input_shape=in_shape, 
        eps=EPSILON, 
        num_epochs=1,
        checkpoint_save_path=None,
        device=device,
        gradient_clip=GRADIENT_CLIP,
        l1_reg_weight=0.0,
        shi_reg_weight=SHI_REG_WEIGHT,
        mtl_ibp_alpha=MTL_IBP_ALPHA,
        pgd_steps=PGD_STEPS,
        pgd_alpha=PGD_ALPHA,
        pgd_restarts=PGD_RESTARTS,
        pgd_alpha_decay_factor=PGD_ALPHA_DECAY_FACTOR,
        pgd_decay_milestones=PGD_DECAY_MILESTONES,
    )
    base_wrapper.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    base_wrapper.eval()
    base_res = base_wrapper.evaluate(eval_loader, test_samples=10000)
    baseline_clean = base_res[0] * 100.0
    baseline_cert = base_res[1] * 100.0
    print(f"Baseline - Clean Acc: {baseline_clean:.4f}%, Certified Acc: {baseline_cert:.4f}%")

    def run_fine_tune(top_k, lambda_reg, lr, num_epochs):
        model = MNIST_ResNet18(num_classes=10).to(device)
        wrapped_model = MTLIBPModelWrapper(
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
            mtl_ibp_alpha=MTL_IBP_ALPHA,
            pgd_steps=PGD_STEPS,
            pgd_alpha=PGD_ALPHA,
            pgd_restarts=PGD_RESTARTS,
            pgd_alpha_decay_factor=PGD_ALPHA_DECAY_FACTOR,
            pgd_decay_milestones=PGD_DECAY_MILESTONES,
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
                    elif isinstance(module, nn.BatchNorm2d):
                        w_squared = module.weight ** 2
                        if module.bias is not None:
                            w_squared += module.bias ** 2
                        penalty += (s * w_squared).sum()
            return penalty

        # Override regularizer
        ctrain.train.certified.mtl_ibp.get_l1_reg = score_weighted_l2_reg

        wrapped_model.train_model(retrain_loader, start_epoch=0, end_epoch=num_epochs)
        wrapped_model.eval()
        res = wrapped_model.evaluate(eval_loader, test_samples=10000)
        clean = res[0] * 100.0
        cert = res[1] * 100.0
        return wrapped_model, clean, cert

    best_top_k = 0.15
    best_lambda_reg = 0.0002
    best_lr = 0.00003

    # Final run: run retraining with best hyperparameters for the full epochs
    best_model, final_clean, final_cert = run_fine_tune(best_top_k, best_lambda_reg, best_lr, num_epochs=NUM_EPOCHS)
    
    print(f"\n--- Final Results ---")
    print(f"Baseline - Clean Acc: {baseline_clean:.4f}%, Certified Acc: {baseline_cert:.4f}%")
    print(f"retrained - Clean Acc: {final_clean:.4f}%, Certified Acc: {final_cert:.4f}%")
    
    print(f"Standard Acc: {final_clean:.4f}% (Baseline: {baseline_clean:.4f}%)")
    print(f"Certified Acc: {final_cert:.4f}% (Baseline: {baseline_cert:.4f}%)")


    # Save the final retrained model
    torch.save(best_model.state_dict(), out_path)
    print(f"retrained model saved to {out_path}")

if __name__ == "__main__":
    main()
