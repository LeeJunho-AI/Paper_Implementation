import torch
import torch.nn as nn

class VGGNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1_layer = nn.Sequential(
       nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride = 2))
    
    self.conv2_layer = nn.Sequential(
       nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride = 2))
    
    self.conv3_layer = nn.Sequential(
       nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride = 2))
    
    self.conv4_layer = nn.Sequential(
       nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride = 2))
    
    self.conv5_layer = nn.Sequential(
       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(kernel_size=2, stride = 2))

    self.fc_layers = nn.Sequential(
       nn.Linear(in_features=512*7*7, out_features=4096),
       nn.ReLU(),
       nn.Linear(in_features=4096, out_features=4096),
       nn.ReLU(),
       nn.Linear(in_features=4096, out_features=10))
    '''
    The originally proposed out_features = 1000 in the paper, 
    but it has been adjusted to 10 to fit the CIFAR-10 dataset.
    '''
    
  def forward(self, x):
    output = self.conv1_layer(x)
    output = self.conv2_layer(output)
    output = self.conv3_layer(output)
    output = self.conv4_layer(output)
    output = self.conv5_layer(output)
    output = output.view(-1, 512*7*7)
    output = self.fc_layers(output)
    return output
  

# Usage example (not executed when imported):
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Quick test of the VGGNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGNet().to(device)
    x = torch.randn(1, 3, 224, 224).to(device)  # Example input tensor
    y = model(x)
    print(f"Output shape: {y.shape}")
