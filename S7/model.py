import torch.nn as nn
import torch.nn.functional as F
class Model_1(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,32,3),  #26, rf=3
                             nn.ReLU(),  
                             nn.Conv2d(32,64,3), #24 , 5
                             nn.ReLU(),
                             nn.Conv2d(64,128,3),  #22 , 7
                             nn.ReLU(), 
                             nn.Conv2d(128,256,3),  #20 , 9
                             nn.ReLU()
                             )
    self.trans1=nn.Sequential(nn.MaxPool2d(2,2),  # 10, 10
                              nn.Conv2d(256,32,1)
                              )
    self.conv2=nn.Sequential(  
                             nn.Conv2d(32,64,3), #8, 14
                             nn.ReLU(),
                             nn.Conv2d(64,128,3),  #6 18
                             nn.ReLU(), 
                             nn.Conv2d(128,256,3),  #4 22
                             nn.ReLU(),
                              nn.Conv2d(256,512,3),  #2 26
                            #  nn.ReLU()
                             )
    self.trans2=nn.Sequential(nn.MaxPool2d(2,2),
                              nn.Conv2d(512,10,1))
  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)

