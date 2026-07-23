import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from ctrain.model_wrappers.sabr_model_wrapper import SABRModelWrapper
from ctrain.data_loaders import load_cifar10

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

class CIFAR_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR_ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
    BATCH_SIZE = 128
    EPSILON = 8 / 255
    NUM_EPOCHS = 160
    LR = 0.0005
    WARM_UP_EPOCHS = 1
    RAMP_UP_EPOCHS = 80
    LR_DECAY_MILESTONES = (120, 140)
    LR_DECAY_FACTOR = 0.2
    SABR_SELECTION_RATIO_LAMBDA = 0.7
    L1_REG_WEIGHT = 0.0
    
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
    
    model = CIFAR_ResNet18(num_classes=10).to(device)
    
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'cifar', 'resnet18_sabr')
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
        device=device,
        sabr_subselection_ratio=SABR_SELECTION_RATIO_LAMBDA,
        l1_reg_weight=L1_REG_WEIGHT
    )

    print("Starting SABR Training...")
    
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
    
    out_path = os.path.join(out_dir, 'resnet18_sabr_8_final.pt')
    torch.save(wrapped_model.state_dict(), out_path)
    print(f"Final model saved to {out_path}")

if __name__ == "__main__":
    main()