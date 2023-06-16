import torch.nn as nn
import torch.nn.functional as F
class Model_1(nn.Module):
  def __init__(self):
    super(Model_1,self).__init__()
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

class Model_2(nn.Module):
  def __init__(self):
    super(Model_2,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,32,3),  #26, rf=3
                             nn.ReLU(),  
                             nn.BatchNorm2d(32),
                             nn.Conv2d(32,64,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(64),
                             nn.Conv2d(64,128,3),  #22 , 7
                             nn.ReLU(), 
                             nn.BatchNorm2d(128),
                             nn.Conv2d(128,256,3),  #20 , 9
                             nn.ReLU()
                             ,nn.BatchNorm2d(256)
                             )
    self.trans1=nn.Sequential(nn.MaxPool2d(2,2),  # 10, 10
                              nn.Conv2d(256,32,1)
                              )
    self.conv2=nn.Sequential(  
                             nn.Conv2d(32,64,3), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(64),
                             nn.Conv2d(64,128,3),  #6 18
                             nn.ReLU(), 
                             nn.BatchNorm2d(128),
                             nn.Conv2d(128,256,3),  #4 22
                             nn.ReLU(),
                             nn.BatchNorm2d(256),
                              nn.Conv2d(256,512,3) #2 26
                             #nn.BatchNorm2d(32)
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

class Model_3(nn.Module):
  def __init__(self):
    super(Model_3,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
              
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                            
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14),
                             
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,8,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(8,12,3), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             
                             nn.Conv2d(12,16,3),  #6, 18
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             nn.MaxPool2d(2,2),  #3 , 19
                             nn.Conv2d(16,16,3) , #1, 27
                             )
    self.trans2=nn.Sequential(
                              nn.Conv2d(16,10,1))
  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)

class Model_4(nn.Module):
  def __init__(self):
    super(Model_4,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
                             nn.Dropout(0.1),
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.1),
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14),
                             nn.Dropout(0.1),
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,8,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(8,12,3), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.1),
                             nn.Conv2d(12,16,3),  #6, 18
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             nn.MaxPool2d(2,2),  #3 , 19
                             nn.Conv2d(16,16,3) , #1, 27
                             )
    self.trans2=nn.Sequential(
                              nn.Conv2d(16,10,1))
  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)

class Model_5(nn.Module):
  def __init__(self):
    super(Model_5,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
                             nn.Dropout(0.5),
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.5),
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14),
                             nn.Dropout(0.5),
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,8,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(8,12,3), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.5),
                             nn.Conv2d(12,16,3),  #6, 18
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             nn.MaxPool2d(2,2),  #3 , 19
                             nn.Conv2d(16,16,3) , #1, 27
                             )
    self.trans2=nn.Sequential(
                              nn.Conv2d(16,10,1))
  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)

class Model_6(nn.Module):
  def __init__(self):
    super(Model_6,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
                             nn.Dropout(0.1),
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             #nn.Dropout(0.5),
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14),
                             #nn.Dropout(0.5),
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,8,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(8,12,3), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.1),
                             nn.Conv2d(12,16,3),  #6, 18
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             nn.MaxPool2d(2,2),  #3 , 19
                             nn.Conv2d(16,16,3) , #1, 27
                             )
    self.trans2=nn.Sequential(nn.Conv2d(16,10,1))

  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)

class Model_7(nn.Module):
  def __init__(self):
    super(Model_7,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
                             nn.Dropout(0.1),
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14)
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,8,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(8,16,3), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             nn.Dropout(0.1),
                             nn.Conv2d(16,16,3),  #6, 18
                            #  nn.ReLU(),
                            #  nn.BatchNorm2d(16),
                             nn.MaxPool2d(2,2),  #3 , 19
                             #nn.Conv2d(16,16,3) , #1, 27
                             )
    self.trans2=nn.Sequential(nn.Conv2d(16,10,1))
    self.gap=nn.AvgPool2d(3)

  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=self.gap(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)
    
class Model_8(nn.Module):
  def __init__(self):
    super(Model_8,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
                             nn.Dropout(0.1),
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14)
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,8,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(8,12,3,padding=1), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.1),
                             nn.Conv2d(12,16,3,padding=1),  #8, 18
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             nn.MaxPool2d(2,2),  #4 , 19
                             nn.Conv2d(16,16,3) , #2, 27
                             )
    self.trans2=nn.Sequential(nn.Conv2d(16,10,1))
    self.gap=nn.AvgPool2d(2)

  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=self.gap(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)
    
class Model_9(nn.Module):
  def __init__(self):
    super(Model_9,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
                             nn.Dropout(0.05),
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14)
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,9,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(9,12,3,padding=1), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.1),
                             nn.Conv2d(12,16,3,padding=1),  #8, 18
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             nn.MaxPool2d(2,2),  #4 , 19
                             nn.Conv2d(16,16,3) , #2, 27
                             )
    self.trans2=nn.Sequential(nn.Conv2d(16,10,1))
    self.gap=nn.AvgPool2d(2)

  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=self.gap(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)

class Model_10(nn.Module):
  def __init__(self):
    super(Model_10,self).__init__()
    self.conv1=nn.Sequential(nn.Conv2d(1,8,3),  #26, rf=3
                             nn.ReLU(),
                             nn.BatchNorm2d(8),
                             nn.Dropout(0.05),
                             nn.Conv2d(8,12,3), #24 , 5
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.MaxPool2d(2,2),  #12 , 6
                             nn.Conv2d(12,14,3),  #10 , 10
                             nn.ReLU(),
                             nn.BatchNorm2d(14)
                             )
    self.trans1=nn.Sequential(
                              nn.Conv2d(14,9,1)
                              )
    self.conv2=nn.Sequential(
                             nn.Conv2d(9,12,3), #8, 14
                             nn.ReLU(),
                             nn.BatchNorm2d(12),
                             nn.Dropout(0.05),
                             nn.Conv2d(12,16,3),  #6, 18
                             nn.ReLU(),
                             nn.BatchNorm2d(16),
                             #nn.MaxPool2d(2,2),  #4 , 19
                             nn.Conv2d(16,16,3) , #4, 27
                             )
    self.trans2=nn.Sequential(nn.Conv2d(16,10,1))
    self.gap=nn.AvgPool2d(4)

  def forward(self,x):
    x=self.conv1(x)
    x=self.trans1(x)
    x=self.conv2(x)
    x=self.trans2(x)
    x=self.gap(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)


