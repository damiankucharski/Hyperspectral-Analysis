import torch.nn.functional as F
from torch import nn
import torch
import tqdm
from sklearn.metrics import accuracy_score


class OurNet(nn.Module):

    def __init__(self,output_units):
        super(OurNet, self).__init__()
        self.conv_1 = torch.nn.Conv3d(1,8,(3,3,7), stride=(1, 1, 1))
        self.conv_2 = torch.nn.Conv3d(8, 16,(3,3,5),stride=(1,1,1))
        self.conv_3 = torch.nn.Conv3d(16,32,(3,3,3), stride=(1, 1, 1))
        self.conv_4 = torch.nn.Conv2d(576,64,(3,3), stride=(1, 1))
        self.flatten = torch.nn.Flatten()
        self.dense_1 = torch.nn.Linear(in_features=18496,out_features=256)
        self.dropout_1 = torch.nn.Dropout(p=0.4)
        self.dense_2 = torch.nn.Linear(in_features=256,out_features=128)
        self.dropout_2 = torch.nn.Dropout(p=0.4)
        self.dense_3 = torch.nn.Linear(in_features=128,out_features=output_units)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.conv_3(x)
        x = F.relu(x)
        x = x.permute((0,2,3,1,4))
        x  =x.reshape((*x.shape[:3],-1))
        x = x.permute((0,3,1,2))
        x = self.conv_4(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.log_softmax(x)
        return x


    def train_model(self, X, y, device, batch_size = 256, epochs = 100, validation_data = [], callbacks = []):

        loss = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        losses_epochs = []
        scores = []
        if validation_data:
            scores_val = []

        for _ in tqdm.tqdm(range(epochs)):
            for idx in range(0,len(y) - batch_size,batch_size):
                optimizer.zero_grad()
                temp_X, temp_y = X[idx:idx+batch_size].to(device), y[idx:idx+batch_size].to(device)
                res = self.forward(temp_X)
                output = loss(res, temp_y.long())
                output.backward()
                optimizer.step()
            losses_epochs.append(output.item())
        
            scores.append(self.score(X,y,device, batch_size))
            if validation_data:
                scores_val.append(self.score(validation_data[0], validation_data[1], device, batch_size))


        if validation_data:
            return losses_epochs, scores, scores_val
        else:
            return losses_epochs, scores
        
    def score(self, X_test , y_test, device, batch_size = 256):
        self.train(False)
        scores = []
        device_cpu = torch.device('cpu')
        with torch.no_grad():
            for idx in tqdm.tqdm(range(0,len(y_test) - batch_size,batch_size)):
                temp_X, temp_y = X_test[idx:idx+batch_size].to(device), y_test[idx:idx+batch_size].to(device)
                res = self(temp_X)
                scores.append(torch.argmax(torch.exp(res.to(device_cpu)),axis=1))
        accuracy_scores = []
        for idx in tqdm.tqdm(range(0,len(y_test) - batch_size,batch_size)):
            accuracy_scores.append(accuracy_score(y_test[idx:idx+batch_size],scores[int(idx/batch_size)]))

        self.train(True)
        return sum(accuracy_scores)/len(accuracy_scores)

