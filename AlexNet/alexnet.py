import torch.nn as nn

class AlexNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1_layer = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 96,
                           kernel_size = 11, stride = 2, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2),
        nn.LocalResponseNorm(size = 5, alpha = 1e-4, beta = 0.75, k = 2))

    self.conv2_layer = nn.Sequential(
        nn.Conv2d(in_channels = 96, out_channels = 256,
                  kernel_size = 5, padding = 3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2),
        nn.LocalResponseNorm(size = 5, alpha = 1e-4, beta = 0.75, k = 2))

    self.conv3_layer = nn.Sequential(
        nn.Conv2d(in_channels = 256, out_channels = 384,
                  kernel_size = 3, stride = 2),
        nn.ReLU())

    self.conv4_layer = nn.Sequential(
        nn.Conv2d(in_channels = 384, out_channels = 384,
                  kernel_size = 3, padding = 1),
        nn.ReLU())

    self.conv5_layer = nn.Sequential(
        nn.Conv2d(in_channels = 384, out_channels = 256,
                  kernel_size = 3, padding = 1),
        nn.ReLU())

    self.fc_layer = nn.Sequential(
        nn.Linear(in_features=256*13*13, out_features = 4096),
        nn.Dropout(0.5),
        nn.Linear(in_features=4096, out_features = 4096),
        nn.Dropout(0.5),
        nn.Linear(in_features=4096, out_features = 10))

        # The paper introduced out_features = 1000 but,
        # I set out_features = 10 for the CIFAR10 dataset.


  def forward(self, x):
    output = self.conv1_layer(x)
    output = self.conv2_layer(output)
    output = self.conv3_layer(output)
    output = self.conv4_layer(output)
    output = self.conv5_layer(output)
    output = output.view(output.size(0), -1)
    output = self.fc_layer(output)
    return output
  

# Usage example (not executed when imported):
if __name__ == "__main__":
    # Quick test of the AlexNet model
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    x = torch.randn(1, 3, 224, 224).to(device)  # Example input tensor
    y = model(x)
    print(f"Output shape: {y.shape}")