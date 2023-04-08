import torch, math 
import torch.nn as nn
import torch.nn.functional as F


#######################################
#                                     #
# THE ATTENTION-BASED ARCHITECTURE    #
#                                     #
#######################################



class MultiheadAttention(nn.Module):
    def __init__(self, heads, sequence_lenght, input_dim):
        super(MultiheadAttention, self).__init__()
        self.heads = heads
        self.sequence_lenght = sequence_lenght
        self.input_dim = input_dim
        self.qkv_dim = input_dim // heads * 3
        
        self.qkv_layer = nn.Linear(in_features=input_dim, out_features=input_dim * 3)
        self.linear_layer = nn.Linear(input_dim, input_dim)
          
    def forward(self, x):
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(-1, self.heads, self.sequence_lenght, self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        values = self.DotProduct(q, k, v)
        values = values.reshape(-1, self.sequence_lenght, self.input_dim)
        return self.linear_layer(values)
        
    def DotProduct(self, q, k, v):
        dk = q.size(-1)
        scaled = torch.matmul(q, torch.transpose(k, 3, 2)) / math.sqrt(dk)
        attention = F.softmax(scaled, dim=3)
        return torch.matmul(attention, v)
    
    
    
class Encoder(nn.Module):
    def __init__(self, heads, sequence_lenght, input_dim):
        super(Encoder, self).__init__()
        self.LayerNorm1 = nn.LayerNorm(input_dim)
        self.LayerNorm2 = nn.LayerNorm(input_dim)
        self.MultiheadAttention = MultiheadAttention(heads, sequence_lenght, input_dim)
        self.feedForward = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim),
            nn.GELU(),
            nn.Dropout1d(0.25),
            nn.Linear(in_features=input_dim, out_features=input_dim)
        )
        
    def forward(self, x):
        x = self.LayerNorm1(x + self.MultiheadAttention(x))
        return x + self.LayerNorm2(self.feedForward(x))
    
    
    
class Transformer(nn.Module):
    def __init__(self, heads, sequence_lenght, input_dim, num_repeat):
        super(Transformer, self).__init__()
        self.layers = self._make_layers(heads, sequence_lenght, input_dim, num_repeat)
        
    def forward(self, x):
        return self.layers(x)
        
    def _make_layers(self, heads, sequence_lenght, input_dim, num_repeat):
        layers = []
        for i in range(num_repeat): 
            layers.append(Encoder(heads, sequence_lenght, input_dim))
        return nn.Sequential(*layers)
    


class VitOCR(nn.Module):
    def __init__(self, heads=8, sequence_lenght=21, input_dim=256, num_repeat=21, num_classes=37):
        super(VitOCR, self).__init__()
        
        self.encoder = Transformer(heads, sequence_lenght, input_dim, num_repeat)
        self.linear = nn.Linear(256, 64)
        self.gru = nn.GRU(input_size=64, hidden_size=256, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
        self.output = nn.Sequential(
            nn.Linear(in_features=256*2, out_features=512),
            nn.Dropout1d(0.25),
            nn.Linear(in_features=512, out_features=num_classes)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        x = self.gru(x)[0]
        x = self.output(x)
        return x



#########################################
#                                       #
# THE CONVOLUTIONAL SIMPLE ARCHITECTURE #
#                                       #
#########################################


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padd=0):
        super(ConvBlock, self).__init__()
        
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        return self.layer(x)
   


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)



class ChannelShuffle(nn.Module):
    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
         # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(ShuffleUnit, self).__init__()
        self.pointWise1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            ChannelShuffle(4)
        )
        self.DWConv = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(reduced_dim),
            nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            # nn.BatchNorm2d(reduced_dim)
        )
        self.pointWise2 = nn.Sequential(
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1, stride=1)
        )
        
    def forward(self, x):
        return x + self.pointWise2(self.DWConv(self.pointWise1(x)))
    


class SeArchitecture(nn.Module):
    def __init__(self, in_channels, num_classes=37):
        super(SeArchitecture, self).__init__()
        self.descriminator = SqueezeExcitation(in_channels, reduced_dim=3)
        self.extractor1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.shuffleUnit = ShuffleUnit(in_channels=64, reduced_dim=128)
        self.extractor2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.linear = nn.Linear(in_features=864, out_features=32)
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
        self.output = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout1d(0.25),
            nn.Linear(in_features=64, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.descriminator(x)
        x = self.extractor1(x)
        x = self.shuffleUnit(x)
        x = self.extractor2(x)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2, end_dim=-1)
        x = self.linear(x)
        x = self.gru(x)[0]
        x = self.output(x)
        return x
    


#########################################
#                                       #
# THE CONVOLUTIONAL DEEPER ARCHITECTURE #
#                                       #
#########################################


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.layer(x)
    


class FPN(nn.Module):
    def __init__(self, cBLock, in_channels):
        super(FPN, self).__init__()
        
        self.layer1 = cBLock(in_channels=in_channels, out_channels=64)
        self.layer2 = cBLock(in_channels=64, out_channels=64)
        self.layer3 = cBLock(in_channels=64, out_channels=64)
        self.layer4 = cBLock(in_channels=64, out_channels=64)
        
        self.pointWise1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.pointWise2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        
        x2 = F.interpolate(self.layer4(x1), size=x1.size()[2:])
        x1 = self.pointWise1(x1) + x2
        x1 = F.interpolate(x1, size=x.size()[2:])
        x = self.pointWise2(x) + x1
        return x



class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(ShuffleUnit, self).__init__()
        self.pointWise1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True)
        )
        self.DWConv = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1),
            nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(reduced_dim)
        )
        self.pointWise2 = nn.Sequential(
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        return x + self.pointWise2(self.DWConv(self.pointWise1(x)))



class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)



class Architecture(nn.Module):
    def __init__(self, num_classes=12):
        super(Architecture, self).__init__()
        self.Pyramid = FPN(ConvBlock, in_channels=3)
        self.ShuffleUnit1 = ShuffleUnit(in_channels=64, reduced_dim=32)
        self.SqueezeExcitation1 = SqueezeExcitation(in_channels=64, reduced_dim=32)
        self.ShuffleUnit2 = ShuffleUnit(in_channels=64, reduced_dim=32)
        self.SqueezeExcitation2 = SqueezeExcitation(in_channels=64, reduced_dim=32)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25),
        )
        self.linear1 = nn.Sequential(
            nn.BatchNorm1d(19),
            nn.Linear(in_features=96, out_features=32),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.gru = nn.GRU(32, 64, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(in_features=64*2, out_features=256),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.Pyramid(x)
        x = self.ShuffleUnit1(x)
        x = self.SqueezeExcitation1(x)
        x = self.ShuffleUnit2(x)
        x = self.SqueezeExcitation2(x)
        x = self.layer1(x)
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(start_dim=2)
        x = self.linear1(x)
        x = self.gru(x)[0]
        x = self.output(x)
        return x