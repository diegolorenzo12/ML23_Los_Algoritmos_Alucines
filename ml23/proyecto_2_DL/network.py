import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()


class SqueezeandExcitationBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeandExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SqueezeandExcitationBlock(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SqueezeandExcitationBlock(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.se3 = SqueezeandExcitationBlock(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer to combat overfitting
        self.dropout = nn.Dropout(0.5)

        # Adaptive Pool layer to flatten the conv output
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.se1(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.se3(x)

        x = self.adap_pool(x)
        x = x.view(x.size(0), -1)  # Flatten layer

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def predict(self, x):
        with torch.inference_mode():
            x = x.to(self.device)
            logits = self.forward(x)
            probas = F.softmax(logits, dim=1)
            return probas

    def save_model(self, model_name: str):
        models_path = file_path / 'models'
        models_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        torch.save(self.state_dict(), models_path / f"{model_name}.pth")

    def load_model(self, model_name: str):
        models_path = file_path / 'models' / f"{model_name}.pth"
        self.load_state_dict(torch.load(models_path, map_location=self.device))