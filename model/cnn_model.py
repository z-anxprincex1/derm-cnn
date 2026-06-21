import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SkinNet(nn.Module):
    """
    A custom CNN architecture designed for skin disease classification.
    Consists of 4 convolutional blocks with Batch Normalization, MaxPooling, and Dropout.
    """
    def __init__(self, num_classes=4):
        super(SkinNet, self).__init__()
        # Input shape: (3, 224, 224)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # After 4 pools, size is 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, 256 * 14 * 14)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNet18Transfer(nn.Module):
    """
    Fine-tuned ResNet-18 model for skin disease classification.
    Uses transfer learning by updating the fully connected layer.
    """
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet18Transfer, self).__init__()
        # Load ResNet-18
        if pretrained:
            # Using modern weights parameter, fall back to pretrained=True if older torchvision
            try:
                from torchvision.models import ResNet18_Weights
                self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            except ImportError:
                self.resnet = models.resnet18(pretrained=True)
        else:
            self.resnet = models.resnet18(pretrained=False)
            
        # Replace the final fully-connected classifier head
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) helper.
    Computes activation heatmaps of the model's final convolutional layer to visualize attention.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap for a given input tensor and class index.
        """
        self.model.eval()
        # Ensure input tensor requires grad
        if not input_tensor.requires_grad:
            input_tensor.requires_grad = True
            
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Calculate Grad-CAM
        # pool gradients across channels
        # gradients shape: [batch, channel, height, width]
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # multiply activations by gradients
        # activations shape: [batch, channel, height, width]
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # apply ReLU to heatmap (keep positive activations only)
        heatmap = torch.clamp(heatmap, min=0)
        
        # normalize between 0 and 1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        return heatmap.cpu().numpy(), output.detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
