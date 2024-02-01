import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 데이터 불러오기
iris_dataset = load_iris()
df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
df['label'] = iris_dataset.target

# 데이터 분할
Y = df['label']
X = df.drop(columns='label')  # X = df.drop(['label'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, random_state=42, stratify=Y)  # Y (df['label']) 분포를 비율에 따라 맞추기


## Data loader ##
BATCH_SIZE = 10

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.int64)
Y_test = torch.tensor(Y_test, dtype=torch.int64)

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


## Network ##
if torch.cuda.is_available():
DEVICE = torch.device('cuda')
else:
DEVICE = torch.device('cpu')

class Net(nn.Module):
def __init__(self):
super(Net, self).__init__()

self.input_layer = nn.Linear(4, 16)
self.hidden_layer1 = nn.Linear(16, 32)
self.hidden_layer2 = nn.Linear(32, 64)
self.output_layer = nn.Linear(64, 3)
self.relu = nn.ReLU()

def forward(self, x):
x = self.input_layer(x)
x = self.relu(x)
x = self.hidden_layer1(x)
x = self.relu(x)
x = self.hidden_layer2(x)
x = self.relu(x)
x = self.output_layer(x)

return x

model = Net().to(DEVICE)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


## Training & Testing ##
EPOCHS = 10

for i in range(EPOCHS):
  print("Epoch {}".format(i+1))

  # Training
  for batch, (X, Y) in enumerate(train_dataloader):
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    prediction = model(X)

    loss = criterion(prediction, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss, current = loss.item(), batch * len(X)

    print(f"Loss: {loss:>7f}, [{current:>5d}]/{len(train_dataloader.dataset):5d}")
  
  # Testing
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, Y in test_dataloader:
      X, Y = X.to(DEVICE), Y.to(DEVICE)

      prediction = model(X)

      test_loss += criterion(prediction, Y).item()
      correct += (prediction.argmax(1) == Y).type(torch.float).sum().item()
  
  test_loss /= len(test_dataloader)
  correct /= len(test_dataloader.dataset)

  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:8f}\n")

  print('-----Complete-----')
