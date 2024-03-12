import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary

class Model10(nn.Module):
    def __init__(self,dropout_value):
        super(Model10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)
        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Model9(nn.Module):
    def __init__(self,dropout_value):
        super(Model9, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Model8(nn.Module):
    def __init__(self,dropout_value):
        super(Model8, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)


class Model7(nn.Module):
    def __init__(self,dropout_value):
        super(Model7, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 28 > 26 | 3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 26 > 24 | 5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) # 24 > 22 | 7

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 22 > 20 | 9

        self.one_cross_one_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        ) # 20 > 20 | 9

        self.trans1 = nn.Sequential(
          nn.MaxPool2d(2,2)
        )   # 20 > 10 | 10

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        ) # 10 > 8 | 14

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14)
        )  # 8 > 6 | 18

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=13, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(13)
        )   #  6 > 4 | 22

        self.gap = nn.AdaptiveAvgPool2d(1)                             # 4 > 1 | 28

        self.one_cross_one_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10)
        )   # 1 > 1 | 28

        self.dropout = nn.Dropout(dropout_value)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.one_cross_one_conv1(x)
        x = self.trans1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dropout(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = self.one_cross_one_conv2(x)
        
        x = x.view(-1, 10)    
        return F.log_softmax(x, dim=-1)

class Model6(nn.Module):
    def __init__(self,dropout_value):
        super(Model6, self).__init__()
        # Input Block 28  
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26 

        # CONVOLUTION BLOCK 1  
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24. 
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22 

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11 
        

        # CONVOLUTION BLOCK 2 4  8
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9 >>> 27
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7 >>> 25

        # OUTPUT BLOCK  remove oradd
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7 >>> 25
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1
        

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.gap(x)
       
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
class Model5(nn.Module):
    def __init__(self,dropout_value):
        super(Model5, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model4(nn.Module):
    def __init__(self,dropout_value):
        super(Model4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class Model3(nn.Module):
    def __init__(self,dropout_value):
        super(Model3, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),

            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),

            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),


            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),


            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.BatchNorm2d(10), NEVER
            # nn.ReLU() NEVER!
        ) # output_size = 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def get_summary(model, input_size) :   
    """
    Generate Model Summary Using torchsummary.

    This function provides a summary of the PyTorch model using the torchsummary library.
    It displays information such as the model architecture, number of parameters,
    and memory consumption.

    Parameters:
    - model (torch.nn.Module): PyTorch model for which the summary is to be generated.
    - input_size (tuple): Tuple representing the input size of the model, e.g., (channels, height, width).
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network = model.to(device)
    return summary(network, input_size=input_size)


# class Model10(nn.Module):
#     def __init__(self):
#         super(Model10, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, 3, padding=0, bias=False)       # 28 > 26 | 3
#         self.batch_norm1 = nn.BatchNorm2d(10)
#         self.conv2 = nn.Conv2d(10, 10, 3, padding=0, bias=False)      # 26 > 24 | 5
#         self.batch_norm2 = nn.BatchNorm2d(10)
#         self.conv3 = nn.Conv2d(10, 10, 3, padding=0, bias=False)      # 24 > 22 | 7
#         self.batch_norm3 = nn.BatchNorm2d(10)
#         self.conv4 = nn.Conv2d(10, 10, 3, padding=0, bias=False)      # 22 > 20 | 9
#         self.batch_norm4 = nn.BatchNorm2d(10)

#         self.point1 = nn.Conv2d(10, 10, 1, padding=0, bias=False)     # 20 > 20 | 9
#         self.batch_normp1 = nn.BatchNorm2d(10)
#         self.pool1 = nn.MaxPool2d(2, 2)                               # 20 > 10 | 10

        
#         self.conv5 = nn.Conv2d(10, 14, 3, padding=0, bias=False)      # 10 > 8 | 14
#         self.batch_norm5 = nn.BatchNorm2d(14)
#         self.conv6 = nn.Conv2d(14, 14, 3, padding=0, bias=False)      #  8 > 6 | 18
#         self.batch_norm6 = nn.BatchNorm2d(14)
#         self.conv7 = nn.Conv2d(14, 13, 3, padding=0, bias=False)      #  6 > 4 | 22
#         self.batch_norm7 = nn.BatchNorm2d(13)


#         self.gap = nn.AdaptiveAvgPool2d(1)                            # 4 > 1 | 28

#         self.point2 = nn.Conv2d(13, 10, 1, padding=0, bias=False)     # 1 > 1 | 28
#         self.batch_normp2 = nn.BatchNorm2d(10)

#         # self.flat = nn.Flatten()
#         # self.fc = nn.Linear(13, 10)

#         # self.dropout = nn.Dropout(0.25)


#     def forward(self, x):
#         x = F.relu(self.batch_norm1(self.conv1(x)))
#         x = F.relu(self.batch_norm2(self.conv2(x)))
#         x = F.relu(self.batch_norm3(self.conv3(x)))
#         #x = self.dropout(x)
#         x = self.batch_norm4(self.conv4(x))

#         x = self.pool1(self.batch_normp1(self.point1(x)))

#         x = F.relu(self.batch_norm5(self.conv5(x)))
#         x = F.relu(self.batch_norm6(self.conv6(x)))
#         #x = self.dropout(x)
#         x = self.batch_norm7(self.conv7(x))

#         x = self.gap(x)

#         x = self.batch_normp2(self.point2(x))
#         # x = self.flat(x)
#         # x = self.fc(x)
#         x = x.view(-1, 10)                                             # 1x1x10 > 10
#         return F.log_softmax(x, dim=-1)