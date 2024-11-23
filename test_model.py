import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import SimpleCNN, Net, ShearTransform
import glob
import pytest
import numpy as np

def get_latest_model():
    model_files = glob.glob('model_*.pth')
    return max(model_files) if model_files else None

def test_model_architecture():
    model = Net()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Model output shape is incorrect"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # Load the latest model
    model_path = get_latest_model()
    assert model_path is not None, "No model file found"
    model.load_state_dict(torch.load(model_path))
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Evaluate accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%"

# New test case 1: Test model predictions are balanced
def test_model_prediction_distribution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load a batch of test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Get predictions for one batch
    data, _ = next(iter(test_loader))
    data = data.to(device)
    with torch.no_grad():
        outputs = model(data)
        predictions = torch.argmax(outputs, dim=1)
    
    # Count predictions for each class
    pred_counts = torch.bincount(predictions)
    # Check if any class has too few or too many predictions
    min_count = len(predictions) * 0.05  # At least 5% of predictions per class
    max_count = len(predictions) * 0.15  # At most 15% of predictions per class
    
    assert torch.all(pred_counts > min_count), "Some classes are underrepresented in predictions"
    assert torch.all(pred_counts < max_count), "Some classes are overrepresented in predictions"

# New test case 2: Test model robustness to augmentations
def test_model_augmentation_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a test image
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    original_image, label = test_dataset[0]
    original_image = original_image.unsqueeze(0).to(device)

    # Apply augmentations
    shear_transform = ShearTransform(shear_range=(-10, 10))
    augmented_image = shear_transform(original_image.squeeze(0))
    augmented_image = augmented_image.unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        original_pred = torch.argmax(model(original_image))
        augmented_pred = torch.argmax(model(augmented_image))

    # Model should be robust to augmentations
    assert original_pred == augmented_pred, "Model predictions change significantly with augmentation"

# New test case 3: Test model confidence
def test_model_confidence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model_path = get_latest_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    confidences = []
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        outputs = model(data)
        probabilities = torch.exp(outputs)  # Convert log_softmax to probabilities
        max_probs = torch.max(probabilities, dim=1)[0]
        confidences.extend(max_probs.cpu().numpy())

    # Calculate average confidence
    avg_confidence = np.mean(confidences)
    assert avg_confidence > 0.8, f"Model average confidence {avg_confidence:.2f} is too low"
    assert avg_confidence < 0.99, f"Model average confidence {avg_confidence:.2f} is suspiciously high"

if __name__ == "__main__":
    pytest.main([__file__]) 