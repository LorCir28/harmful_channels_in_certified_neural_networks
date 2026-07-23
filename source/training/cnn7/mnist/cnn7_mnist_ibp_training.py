import os
import sys
import torch
from torch.utils.data import DataLoader

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.model_wrappers import ShiIBPModelWrapper
from ctrain.data_loaders import load_mnist
from ctrain.model_definitions.models_shi import CNN7_Shi

# ==========================================
# HYPERPARAMETERS & MACROS
# ==========================================
BATCH_SIZE = 256
EPSILON = 0.1
NUM_EPOCHS = 70
LR = 0.0005
WARM_UP_EPOCHS = 1
RAMP_UP_EPOCHS = 20
LR_DECAY_MILESTONES = (50, 60)
LR_DECAY_FACTOR = 0.2
L1_REG_WEIGHT = 1e-5
GRADIENT_CLIP = 10
SHI_REG_WEIGHT = 0.5 # called lambda in the mtl-ibp paper

def main():
    print("Loading MNIST data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    print(f"Using {len(train_loader.dataset)} samples for IBP training.")

    in_shape = (1, 28, 28)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the CNN7 architecture from Shi et al.
    model = CNN7_Shi(in_shape=in_shape, n_classes=10).to(device)
    
    # Wrap model for SHI-IBP training
    wrapped_model = ShiIBPModelWrapper(
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
        shi_reg_weight=SHI_REG_WEIGHT
    )

    print("Starting IBP Training...")
    wrapped_model.train_model(train_loader)
    
    # Save the model
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'cnn7', 'mnist')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cnn7_mnist_ibp_1.pt')
    torch.save(wrapped_model.state_dict(), out_path)
    print(f"Model saved to {out_path}")
    
    # Evaluate
    print("Evaluating Model...")
    std_acc, cert_acc, adv_acc = wrapped_model.evaluate(test_loader)
    print(f"Standard Accuracy: {std_acc}")
    print(f"Certified Accuracy: {cert_acc}")

if __name__ == "__main__":
    main()
