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
from ctrain.data_loaders import load_mnist
from source.training.resnet18.mnist.resnet18_mnist_sabr_training import MNIST_ResNet18

# Prevent re-initializing the weights of loaded model in the training loop
import ctrain.train.certified.sabr
ctrain.train.certified.sabr.ibp_init_shi = lambda *args, **kwargs: None
ctrain.train.certified.sabr.save_checkpoint = lambda *args, **kwargs: None

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
    # Hyperparameters
    BATCH_SIZE = 256
    EPSILON = 0.3
    NUM_EPOCHS = 10
    LR = 1e-5
    LAMBDA_REG = 1e-4
    TOP_K = 0.05
    
    # SABR specific parameters from training script
    GRADIENT_CLIP = 10
    SHI_REG_WEIGHT = 0.5
    SABR_SUBSELECTION_RATIO = 0.6
    PGD_STEPS = 8
    PGD_ALPHA = 0.5
    PGD_RESTARTS = 1
    PGD_EARLY_STOPPING = False
    PGD_ALPHA_DECAY_FACTOR = 0.1
    PGD_DECAY_MILESTONES = (4, 7)
    PGD_EPS_FACTOR = 1.0

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
    json_path = os.path.join(PROJECT_ROOT, 'jsons', 'resnet18', 'mnist', 'resnet18_mnist_sabr_scores.json')
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

    # 3. Construct model and wrapper
    model = MNIST_ResNet18(num_classes=10).to(device)
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'mnist')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'resnet18_mnist_sabr_l2.pt')

    wrapped_model = SABRModelWrapper(
        model=model, 
        input_shape=[1, 28, 28], 
        eps=EPSILON, 
        num_epochs=NUM_EPOCHS,
        lr=LR,
        warm_up_epochs=0,
        ramp_up_epochs=0,
        lr_decay_factor=0.5,
        lr_decay_milestones=(5, 8),
        checkpoint_save_path=out_dir,
        device=device,
        l1_reg_weight=LAMBDA_REG,
        gradient_clip=GRADIENT_CLIP,
        shi_reg_weight=SHI_REG_WEIGHT,
        sabr_subselection_ratio=SABR_SUBSELECTION_RATIO,
        pgd_steps=PGD_STEPS,
        pgd_alpha=PGD_ALPHA,
        pgd_restarts=PGD_RESTARTS,
        pgd_early_stopping=PGD_EARLY_STOPPING,
        pgd_alpha_decay_factor=PGD_ALPHA_DECAY_FACTOR,
        pgd_decay_milestones=PGD_DECAY_MILESTONES,
        pgd_eps_factor=PGD_EPS_FACTOR
    )

    # 4. Load pre-trained weights
    pretrained_path = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'mnist', 'resnet18_mnist_sabr_3.pt')

    print(f"Loading pretrained weights from {pretrained_path}...")
    wrapped_model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

    # Make sure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    # 5. Evaluate baseline accuracies dynamically
    print("\n--- Evaluating Pre-trained Original Model on entire test set ---")
    wrapped_model.eval()
    orig_res = wrapped_model.evaluate(eval_loader, test_samples=10000)
    baseline_clean = orig_res[0] * 100.0
    baseline_cert = orig_res[1] * 100.0
    print(f"Original Model - Clean Acc: {baseline_clean:.4f}%, Certified Acc: {baseline_cert:.4f}%")

    # Generate scores map for this top_k
    device_scores_map = get_device_scores_map(wrapped_model, pos_attributions, TOP_K, max_pos_score, device)

    # Define custom L2 penalty function
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
    ctrain.train.certified.sabr.get_l1_reg = score_weighted_l2_reg

    # 6. Start re-training
    print("\n--- Starting re-training with score-weighted L2 regularization ---")
    print(f"TOP_K={TOP_K}, LAMBDA_REG={LAMBDA_REG}, LR={LR}")
    for epoch in range(0, NUM_EPOCHS, 5):
        end_epoch = min(epoch + 5, NUM_EPOCHS)
        wrapped_model.train_model(retrain_loader, start_epoch=epoch, end_epoch=end_epoch)
        
        print(f"--- Evaluating after Epoch {end_epoch} ---")
        wrapped_model.eval()
        res = wrapped_model.evaluate(eval_loader, test_samples=10000)
        print(f"Clean Acc: {res[0]*100.0:.4f}%, Certified Acc: {res[1]*100.0:.4f}%")

    # 7. Final evaluation
    print("\n--- Final Evaluation of re-trained Model on entire test set ---")
    wrapped_model.eval()
    final_res = wrapped_model.evaluate(eval_loader, test_samples=10000)
    final_clean = final_res[0] * 100.0
    final_cert = final_res[1] * 100.0
    print(f"re-trained Model - Clean Acc: {final_clean:.4f}%, Certified Acc: {final_cert:.4f}%")

    print(f"Standard Acc: {final_clean:.4f}% (Baseline: {baseline_clean:.4f}%)")
    print(f"Certified Acc: {final_cert:.4f}% (Baseline: {baseline_cert:.4f}%)")

    # Save the final fine-tuned model
    torch.save(wrapped_model.state_dict(), out_path)
    print(f"re-trained model saved to {out_path}")

if __name__ == "__main__":
    main()
