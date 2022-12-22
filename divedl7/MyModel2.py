from torch import nn

def LeNet_9():
    """LeNet_9 Model"""
    def __init__(self):
        self.net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=9, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=9), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 2 * 2, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))