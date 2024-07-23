import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


x= torch.randn(100,10)
y=torch.randn(100,1)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(10,50)
        self.fc2=nn.Linear(50,20)
        self.fc3=nn.Linear(20,1)


    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return x


model=SimpleNet()        

criterion=nn.MSELoss()
optimizer= optim.Adam(model.parameters(),lr=0.001)

for i in range(10):
    outputs=model(x)
    loss=criterion(outputs,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f" loss :  { loss.item()}")

