import torch
import pandas as pd
import numpy as np

# Dataset Class
class csvData(torch.utils.data.Dataset):
    #load csv into pandas dataframe
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        # (Rows, Columns) of data
        self.shape = self.data.shape
        # split into x (inputs) and y (outputs)
        self.x = self.data.values[:, self.shape[1] - 6:self.shape[1] - 1].astype(float)
        self.y = self.data.values[:, self.shape[1] - 1:].astype(int)

    def __getitem__(self, index):
        value = self.start + index * self.step
        assert value < self.end
        return value
    
#Logistic Regression Class
class LogisticRegression(torch.nn.Module):
    def __init__(self):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(5, 1)
         
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred        
    
auditData = csvData("audit_risk.csv")
LRmodel = LogisticRegression()

validation_split = .2
shuffle_dataset = True
random_seed= 24

# Creating data indices for training and validation splits
dataset_size = auditData.shape[0]
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Setting up the Loss Function
criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
# Optimization Function
optimizer = torch.optim.Rprop(LRmodel.parameters(), lr=0.01)

# Training the Model
X = torch.Tensor(auditData.x[train_indices])
Y = torch.Tensor(auditData.y[train_indices])

for epoch in range(100):
    LRmodel.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = LRmodel(X)
    # Compute Loss
    loss = criterion(y_pred, Y)
    # Backward pass
    loss.backward()
    optimizer.step()
    
# Get Prediction
X = torch.Tensor(auditData.x[val_indices])
Y = torch.Tensor(auditData.y[val_indices])
correct = 0

for i in range(len(val_indices)):
    actual = int(round(Y[i].item()))
    predict = int(round(LRmodel(X[i]).item()))
    print(predict, "  ", actual)
    
    if predict == actual:
        correct = correct +1
        
print(correct / len(val_indices))