import torch
import torch.nn as nn
import torch.nn.functional as F

class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,24, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(24,36, kernel_size=5, stride=2),
            nn.ELU(),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),  
            nn.ELU(),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),  
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1), 
            nn.ELU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*1*18,1164),
            nn.ELU(),

            nn.Linear(1164,100),
            nn.ELU(),

            nn.Linear(100,50),
            nn.ELU(),

            nn.Linear(50,10),
            nn.ELU(),

            nn.Linear(10,1),
        )

    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
# if __name__ == "__main__":
#     model = NvidiaModel()
#     dummy = torch.randn(2,3,66,200)
#     output = model(dummy)
#     print(output.shape)