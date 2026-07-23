import os
import sys
import torch
from torch.utils.data import DataLoader

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.model_wrappers.sabr_model_wrapper import SABRModelWrapper
from ctrain.data_loaders import load_mnist
from ctrain.model_definitions.models_shi import CNN7_Shi

# ==========================================
# HYPERPARAMETERS & MACROS
# ==========================================
BATCH_SIZE = 256
EPSILON = 0.3  # Standard Linf epsilon for MNIST
NUM_EPOCHS = 70
LR = 0.0005
WARM_UP_EPOCHS = 1
RAMP_UP_EPOCHS = 20
LR_DECAY_MILESTONES = (50, 60)
LR_DECAY_FACTOR = 0.2
L1_REG_WEIGHT = 1e-6
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
CHECKPOINT_SAVE_INTERVAL = 10

def main():
    print("Loading MNIST data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    print(f"Using {len(train_loader.dataset)} samples for SABR training.")

    in_shape = [1, 28, 28]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'mnist', 'cnn7_mnist_sabr'))
    os.makedirs(out_dir, exist_ok=True)
    
    model = CNN7_Shi(in_shape=tuple(in_shape), n_classes=10).to(device)
    
    # Wrap model for SABR training
    wrapped_model = SABRModelWrapper(
        model=model, 
        input_shape=in_shape, 
        eps=EPSILON, 
        num_epochs=NUM_EPOCHS, 
        device=device,
        lr=LR,
        warm_up_epochs=WARM_UP_EPOCHS,
        ramp_up_epochs=RAMP_UP_EPOCHS,
        lr_decay_milestones=LR_DECAY_MILESTONES,
        lr_decay_factor=LR_DECAY_FACTOR,
        l1_reg_weight=L1_REG_WEIGHT,
        gradient_clip=GRADIENT_CLIP,
        shi_reg_weight=SHI_REG_WEIGHT,
        sabr_subselection_ratio=SABR_SUBSELECTION_RATIO,
        pgd_steps=PGD_STEPS,
        pgd_alpha=PGD_ALPHA,
        pgd_restarts=PGD_RESTARTS,
        pgd_early_stopping=PGD_EARLY_STOPPING,
        pgd_alpha_decay_factor=PGD_ALPHA_DECAY_FACTOR,
        pgd_decay_milestones=PGD_DECAY_MILESTONES,
        pgd_eps_factor=PGD_EPS_FACTOR,
        checkpoint_save_path=out_dir,
        checkpoint_save_interval=CHECKPOINT_SAVE_INTERVAL
    )

    print("Starting SABR Training...")
    wrapped_model.train_model(train_loader=train_loader, val_loader=test_loader)

    print("\n--- Evaluating Final Model ---")
    wrapped_model.eval()
    
    eval_results = wrapped_model.evaluate(test_loader, eval_method='IBP')
    
    # Unpack safely depending on what the wrapper returns
    std_acc = eval_results[0]
    cert_acc = eval_results[1]
    
    print(f"Final Clean Acc: {std_acc * 100.0:.2f}%")
    print(f"Final Cert Acc: {cert_acc * 100.0:.2f}%")
    if len(eval_results) > 2:
        print(f"Final Adv Acc: {eval_results[2] * 100.0:.2f}%")

    out_path = os.path.join(out_dir, 'cnn7_mnist_sabr.pt')
    print(f"Saving final weights to {out_path}")
    torch.save(wrapped_model.state_dict(), out_path)
    
    # Also save to the main model_weights folder to maintain consistency
    out_dir_main = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'mnist')
    os.makedirs(out_dir_main, exist_ok=True)
    out_path_main = os.path.join(out_dir_main, 'cnn7_mnist_sabr_3.pt')
    torch.save(wrapped_model.state_dict(), out_path_main)
    print(f"Also saved to {out_path_main}")
    print("Done!")

if __name__ == '__main__':
    main()