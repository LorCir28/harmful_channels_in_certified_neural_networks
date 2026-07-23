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


from ctrain.model_wrappers.crown_ibp_model_wrapper import CrownIBPModelWrapper
from ctrain.data_loaders import load_cifar10

# Define Custom ResNet18 to ensure explicit ReLUs
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
    # RICORDARSI DI MODIFICARE OUTPUTTED MODEL WEIGHTS FILE NAME
    
    print("Loading CIFAR-10 data...")
    data_root = os.path.join(PROJECT_ROOT, 'data')
    train_loader_full, test_loader_full = load_cifar10(batch_size=BATCH_SIZE, val_split=False, data_root=data_root)

    # Split the train set: First 40000 for IBP Training
    train_ds = train_loader_full.dataset
    ibp_train_ds = Subset(train_ds, range(0, 40000))
    train_loader = DataLoader(ibp_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
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
        
    print(f"Using {len(ibp_train_ds)} samples for CROWN-IBP training.")

    in_shape = [3, 32, 32]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CIFAR_ResNet18(num_classes=10).to(device)
    
    # Wrap model for CROWN-IBP training
    wrapped_model = CrownIBPModelWrapper(
        model=model, 
        input_shape=in_shape, 
        eps=EPSILON, 
        num_epochs=NUM_EPOCHS, 
        device=device
    )

    print("Starting CROWN-IBP Training...")
    wrapped_model.train_model(train_loader)
    
    # Save the model
    out_dir = os.path.join(PROJECT_ROOT, 'model_weights', 'resnet18', 'cifar')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'resnet18_crown_ibp_2.pt')
    torch.save(wrapped_model.state_dict(), out_path)
    print(f"Model saved to {out_path}")
    
    # Evaluate
    print("Evaluating Model...")
    eval_results = wrapped_model.evaluate(test_loader, eval_method='SABR')
    print(f"Standard Accuracy: {eval_results[0]}")
    print(f"Certified Accuracy: {eval_results[1]}")
    if len(eval_results) > 2:
        print(f"Adversarial Accuracy: {eval_results[2]}")

if __name__ == "__main__":
    main()