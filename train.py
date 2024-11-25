import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
import random

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(20 * 5 * 5, 128)  # 20 channels * 5 * 5 spatial dimensions
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # After pool: 13x13
        x = self.pool(self.relu(self.conv2(x)))  # After pool: 5x5 
        x = x.view(-1, 20 * 5 * 5)  # Flatten: 20 channels * 5 * 5 spatial dimensions
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            
            
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            
            
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class ShearTransform:
    def __init__(self, shear_range=(-10, 10)):
        self.shear_range = shear_range
        
    def __call__(self, img):
        shear = random.uniform(self.shear_range[0], self.shear_range[1])
        return TF.affine(img, 
                        angle=0,  # no rotation
                        translate=[0, 0],  # no translation
                        scale=1.0,  # no scaling
                        shear=[shear, 0.0],  # shear only in x-direction
                        interpolation=TF.InterpolationMode.BILINEAR,
                        fill=1)  # fill with white

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Updated transform pipeline with shear
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        ShearTransform(shear_range=(-10, 10)),  # Add shear transform
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

def visualize_augmentations():
    save_dir = 'augmentation_samples'
    os.makedirs(save_dir, exist_ok=True)
    
    # Updated transform pipeline with shear
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        ShearTransform(shear_range=(-10, 10)),  # Add shear transform
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    dataset = datasets.MNIST('data', train=True, download=True, transform=None)
    
    # Create figure for visualization
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))  # Changed to 3 rows to show more variations
    
    for idx in range(5):
        # Get original image
        image, label = dataset[idx]
        
        # Convert to tensor for saving
        original_tensor = transforms.ToTensor()(image)
        
        # Apply transformations twice to show variation
        augmented_image1 = transform(image)
        augmented_image2 = transform(image)
        
        # Denormalize for visualization
        augmented_display1 = augmented_image1.clone()
        augmented_display1 = augmented_display1 * 0.3081 + 0.1307
        
        augmented_display2 = augmented_image2.clone()
        augmented_display2 = augmented_display2 * 0.3081 + 0.1307
        
        # Plot original
        axes[0][idx].imshow(image, cmap='gray')
        axes[0][idx].set_title(f'Original (Label: {label})')
        axes[0][idx].axis('off')
        
        # Plot two different augmentations
        axes[1][idx].imshow(augmented_display1[0], cmap='gray')
        axes[1][idx].set_title('Augmented 1')
        axes[1][idx].axis('off')
        
        axes[2][idx].imshow(augmented_display2[0], cmap='gray')
        axes[2][idx].set_title('Augmented 2')
        axes[2][idx].axis('off')
        
        # Save individual images
        save_image(original_tensor, f'{save_dir}/original_{idx}.png')
        save_image(augmented_image1, f'{save_dir}/augmented1_{idx}.png')
        save_image(augmented_image2, f'{save_dir}/augmented2_{idx}.png')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/augmentation_comparison.png')
    plt.close()
    
    print(f"Augmentation samples saved in '{save_dir}' directory")

if __name__ == "__main__":
    visualize_augmentations()
    train() 