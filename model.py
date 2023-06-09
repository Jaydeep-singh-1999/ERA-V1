
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
  def __init__(self):
     super(Net, self).__init__()
     self.conv1=nn.Sequential(nn.Conv2d(1,16,3,padding=1),  #rf=3, 28*28 
                                 nn.ReLU(),
                                 nn.BatchNorm2d(16),
                                 nn.Conv2d(16,32,3,padding=1), #rf=5, 28
                                 nn.ReLU(),
                                 nn.BatchNorm2d(32),
                                 nn.MaxPool2d(2,2),  #rf=6  14
                                 ) 
     self.trans1=nn.Conv2d(32,10,1)
     self.conv2=nn.Sequential(nn.Conv2d(10,16,3,padding=1), #rf=10, 14*14
                              nn.ReLU(),
                              nn.BatchNorm2d(16),
                              nn.Dropout(0.05),   
                              nn.Conv2d(16,32,3,padding=1), #rf=14, 14
                              nn.ReLU(),
                              nn.BatchNorm2d(32), 
                              nn.MaxPool2d(2,2)  #Rf=15 , 7
                              )  
     self.trans2=nn.Conv2d(32,10,1)
     #self.gap=nn.AvgPool2d(4)
     self.conv3=nn.Sequential(nn.Conv2d(10,16,3,padding=1), #rf=23, 7
                              nn.ReLU()
                              ,nn.BatchNorm2d(16), 
                              nn.Dropout(0.05), 
                              nn.Conv2d(16,32,2,padding=1), #rf=31, 7
                               nn.ReLU()
                              ,nn.BatchNorm2d(32), 
                              #nn.MaxPool2d(2,2) #3
                              )
     self.trans3=nn.Conv2d(32,10,1)
     self.conv4=nn.Sequential(nn.Conv2d(10,16,3,padding=1), #rf=39, 7
                              nn.ReLU()
                              ,nn.BatchNorm2d(16), 
                              nn.Dropout(0.05), 
                              nn.Conv2d(16,32,2,padding=1), #rf=47, 7
                               nn.ReLU()
                              ,nn.BatchNorm2d(32), 
                              nn.MaxPool2d(2,2) #rf=48, 3
                              )
     
     self.gap=nn.AvgPool2d(3)
     self.trans4=nn.Conv2d(32,10,1)

     

  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
   
    x=self.conv3(x)
    x=self.trans3(x)
    x=self.conv4(x)
    
    x=self.gap(x)
    x=self.trans4(x)
    x=x.view(-1,10)
    #x=x.fc(x)

    return F.log_softmax(x)


def Model_summary(model):
    return summary(model,input_size=(1,28,28))

