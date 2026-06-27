from torch import nn

class LeNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),  # Why 16*5*5 ?
            nn.ReLU(), 
            nn.Linear(120, 84), # (N, 120) -> (N, 84)
            nn.ReLU(),
            nn.Linear(84, 10)   # (N, 84)  -> (N, 10)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_scores = self.classifier(features)
        return class_scores