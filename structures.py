from collections import namedtuple
import torch
from torch.utils.data import Dataset


data_split_tuple = namedtuple('data_split_tuple', ['label','n_clusters','chosen_clusters'])
train_test_tuple = namedtuple('train_test_tuple', ['label','train_set','test_set'])

class HyperDataset(Dataset):

  def __init__(self, train_test_tuples, train = True):
    self.y = []
    self.X = []
    self.train = train
    for train_test_tuple in train_test_tuples:
      X_temp = train_test_tuple.train_set if train else train_test_tuple.test_set
      self.X.append(X_temp)
      self.y.extend([train_test_tuple.label for i in range(X_temp.shape[0])])
    self.y = torch.Tensor(self.y)
    self.X = torch.cat(self.X)
  
  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, ndx):
    return tuple([self.X[ndx,:], self.y[ndx]])

class ZeroSet(Dataset):


  def __init__(self, train_set, test_set, zero_label, train = True):

    self.y = torch.Tensor(train_set.X.shape[0] * [0] + test_set.X.shape[0] * [1])
    self.X = torch.cat([train_set.X, test_set.X])
    indices = [i for i in range(len(self.y))]
    random.seed(200)
    random.shuffle(indices)
    self.y = self.y[indices]
    self.X = self.X[indices, :]
    length = len(self.y)
    if train:
      self.y = self.y[:int(length * 0.75)]
      self.X = self.X[:int(length * 0.75), :]
    else:
      self.y = self.y[int(length * 0.75):]
      self.X = self.X[int(length * 0.75):, :]
    
  def __len__(self):
    return len(self.y)

  def __getitem__(self, ndx):
    return (self.X[ndx,:], self.y[ndx])

  