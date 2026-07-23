import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.model_wrappers.mtl_ibp_model_wrapper import MTLIBPModelWrapper
from ctrain.data_loaders import load_mnist

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
GRADIENT_CLIP = 10
L1_REG_WEIGHT = 1e-6
SHI_REG_WEIGHT = 0.5 
MTL_IBP_ALPHA = 0.08
PGD_STEPS = 8
PGD_EARLY_STOPPING = False
PGD_ALPHA = 0.5
PGD_RESTARTS = 1
PGD_ALPHA_DECAY_FACTOR = 0.1
PGD_DECAY_MILESTONES = (4, 7)
CHECKPOINT_SAVE_INTERVAL = 10


# pgd_steps (int): Number of PGD steps for adversrial loss computation.
# pgd_alpha (float): PGD step size for adversarial loss calculation.
# pgd_restarts (int): Number of PGD restarts for adversarial loss calculation.
# pgd_alpha_decay_factor (float): PGD alpha decay factor.
# pgd_decay_milestones (tuple): Milestones for PGD alpha decay.

# Define Custom ResNet18 for MNIST (1 input channel) to ensure explicit ReLUs
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

        # Changed to 1 input channel for MNIST
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

def main():
    print("Loading MNIST data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    print(f"Using {len(train_loader.dataset)} samples for MTL IBP training.")
    print(f"Using {len(test_loader.dataset)} samples for evaluation.")

    in_shape = [1, 28, 28]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MNIST_ResNet18(num_classes=10).to(device)
    
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'mnist', 'resnet18_mnist_mtl_ibp')
    os.makedirs(out_dir, exist_ok=True)
    
    # Wrap model for MTL-IBP training
    wrapped_model = MTLIBPModelWrapper(
        model=model, 
        input_shape=in_shape, 
        eps=EPSILON, 
        num_epochs=NUM_EPOCHS,
        lr=LR,
        warm_up_epochs=WARM_UP_EPOCHS,
        ramp_up_epochs=RAMP_UP_EPOCHS,
        lr_decay_factor=LR_DECAY_FACTOR,
        lr_decay_milestones=LR_DECAY_MILESTONES,
        checkpoint_save_path=out_dir,
        device=device,
        gradient_clip=GRADIENT_CLIP,
        l1_reg_weight=L1_REG_WEIGHT,
        shi_reg_decay=SHI_REG_WEIGHT,
        mtl_ibp_alpha=MTL_IBP_ALPHA,
        pgd_steps=PGD_STEPS,
        pgd_early_stopping=PGD_EARLY_STOPPING,
        pgd_alpha=PGD_ALPHA,
        pgd_restarts=PGD_RESTARTS,
        pgd_alpha_decay_factor=PGD_ALPHA_DECAY_FACTOR,
        pgd_decay_milestones=PGD_DECAY_MILESTONES,
        checkpoint_save_interval=CHECKPOINT_SAVE_INTERVAL
    )

    print("Starting MTL IBP Training...")
    
    # Train and evaluate in chunks of 10 epochs
    for epoch in range(0, NUM_EPOCHS, 10):
        end_epoch = min(epoch + 10, NUM_EPOCHS)
        wrapped_model.train_model(train_loader, start_epoch=epoch, end_epoch=end_epoch)
        
        print(f"--- Evaluating after Epoch {end_epoch} ---")
        wrapped_model.eval()
        res = wrapped_model.evaluate(test_loader)
        std_acc = res[0]
        cert_acc = res[1]
        adv_acc = res[2] if len(res) > 2 else "N/A"
        print(f"Standard Accuracy: {std_acc:.4f}")
        print(f"Certified Accuracy: {cert_acc:.4f}")
        if adv_acc != "N/A":
            print(f"Adversarial Accuracy: {adv_acc:.4f}")
    
    out_path = os.path.join(out_dir, 'resnet18_mnist_mtl_ibp_3.pt')
    torch.save(wrapped_model.state_dict(), out_path)
    print(f"Final model saved to {out_path}")

if __name__ == "__main__":
    main()