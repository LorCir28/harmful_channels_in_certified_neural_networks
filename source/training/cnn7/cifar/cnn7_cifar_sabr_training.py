import os
import sys
import torch
from torch.utils.data import DataLoader, Subset

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.model_wrappers.sabr_model_wrapper import SABRModelWrapper
from ctrain.model_definitions.models_shi import CNN7_Shi
from ctrain.data_loaders import load_cifar10

def main():
    BATCH_SIZE = 128
    EPSILON = 2 / 255
    NUM_EPOCHS = 160
    LR = 0.0005
    WARM_UP_EPOCHS = 1
    RAMP_UP_EPOCHS = 80
    LR_DECAY_MILESTONES = (120, 140)
    LR_DECAY_FACTOR = 0.2
    SABR_SELECTION_RATIO_LAMBDA = 0.1
    L1_REG_WEIGHT = 1e-6
    RELU_SHRINKAGE = 0.8
    
    print("Loading CIFAR-10 data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader_full, test_loader_full = load_cifar10(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    # Use entire train set (50000 samples)
    train_ds = train_loader_full.dataset
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Use remaining 9000 samples of test set for testing
    test_ds = test_loader_full.dataset
    eval_ds = Subset(test_ds, range(1000, 10000))
    test_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    if hasattr(test_loader_full, 'normalised'):
        test_loader.normalised = test_loader_full.normalised
        test_loader.mean = getattr(test_loader_full, 'mean', None)
        test_loader.std = test_loader_full.std
        test_loader.min = test_loader_full.min
        test_loader.max = test_loader_full.max
    
    # Copy necessary custom attributes from the original loader for the wrapper
    if hasattr(train_loader_full, 'normalised'):
        train_loader.normalised = train_loader_full.normalised
        train_loader.mean = getattr(train_loader_full, 'mean', None)
        train_loader.std = train_loader_full.std
        train_loader.min = train_loader_full.min
        train_loader.max = train_loader_full.max
        
    print(f"Using {len(train_ds)} samples for SABR training.")
    print(f"Using {len(eval_ds)} samples for evaluation.")

    in_shape = [3, 32, 32]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CNN7_Shi(in_shape=(3, 32, 32), width=64, linear_size=512, n_classes=10).to(device)
    
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'cifar', 'cnn7_cifar_sabr_2')
    os.makedirs(out_dir, exist_ok=True)
    
    # Wrap model for SABR training
    wrapped_model = SABRModelWrapper(
        model=model, 
        input_shape=in_shape, 
        eps=EPSILON, 
        num_epochs=NUM_EPOCHS,
        lr=LR,
        warm_up_epochs=WARM_UP_EPOCHS,
        ramp_up_epochs=RAMP_UP_EPOCHS,
        lr_decay_milestones=LR_DECAY_MILESTONES,
        lr_decay_factor=LR_DECAY_FACTOR,
        checkpoint_save_path=out_dir,
        checkpoint_save_interval=10,
        device=device,
        sabr_subselection_ratio=SABR_SELECTION_RATIO_LAMBDA,
        l1_reg_weight=L1_REG_WEIGHT,
        relu_shrinkage=RELU_SHRINKAGE
    )

    print("Starting SABR Training...")
    
    wrapped_model.train_model(train_loader)
        
    print(f"--- Evaluating after Training ---")
    wrapped_model.eval()
    res = wrapped_model.evaluate(test_loader)
    std_acc = res[0]
    cert_acc = res[1]
    adv_acc = res[2] if len(res) > 2 else "N/A"
    print(f"Standard Accuracy: {std_acc:.4f}")
    print(f"Certified Accuracy: {cert_acc:.4f}")
    if adv_acc != "N/A":
        print(f"Adversarial Accuracy: {adv_acc:.4f}")
    
    out_path = os.path.join(out_dir, 'cnn7_cifar_sabr_2.pt')
    torch.save(wrapped_model.state_dict(), out_path)
    print(f"Final model saved to {out_path}")

if __name__ == "__main__":
    main()