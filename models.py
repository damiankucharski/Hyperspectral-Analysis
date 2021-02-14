import torch

class ModelSimple(torch.nn.Module):

  def __init__(self):
    super(ModelSimple, self).__init__()
    self.fc1 = torch.nn.Linear(25*25*30, 1000)
    self.fc2 = torch.nn.Linear(1000,100)
    self.fc3 = torch.nn.Linear(100, 1)
  
  def forward(self, X):
    X = X.view(-1, 25*25*30)
    X = torch.relu(self.fc1(X))
    X = torch.relu(self.fc2(X))
    X = torch.sigmoid(self.fc3(X))
    return X
