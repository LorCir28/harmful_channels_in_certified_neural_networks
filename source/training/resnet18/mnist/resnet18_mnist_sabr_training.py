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


from ctrain.model_wrappers.sabr_model_wrapper import SABRModelWrapper
from ctrain.data_loaders import load_mnist
from auto_LiRPA import BoundedTensor, PerturbationLpNorm

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

    print(f"Using {len(train_loader.dataset)} samples for SABR training.")

    in_shape = [1, 28, 28]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'mnist', 'resnet18_mnist_sabr'))
    os.makedirs(out_dir, exist_ok=True)
    
    model = MNIST_ResNet18(num_classes=10).to(device)
    
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

    print("\\n--- Evaluating Final Model ---")
    wrapped_model.eval()
    clean_acc, cert_acc = wrapped_model.evaluate(test_loader, eval_method='IBP')
    print(f"Final Clean Acc: {clean_acc * 100.0:.2f}%")
    print(f"Final Cert Acc: {cert_acc * 100.0:.2f}%")

    out_path = os.path.join(out_dir, 'resnet18_mnist_sabr.pt')
    print(f"Saving final weights to {out_path}")
    torch.save(wrapped_model.state_dict(), out_path)
    print("Done!")

if __name__ == '__main__':
    main()
