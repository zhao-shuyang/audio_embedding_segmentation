import torch
import torch.nn as nn
import torch.nn.functional as Fx
import torch.nn.init as init
import torch.nn.utils.rnn
from torch import optim

class EmbNet(nn.Module):
    def __init__(self):
        super(EmbNet,self).__init__() 
        self.layer1a = nn.Sequential(nn.Conv2d(1,32,kernel_size=5,padding=2),nn.BatchNorm2d(32))
        self.layer1b = nn.Sequential(nn.Conv2d(16,32,kernel_size=5,padding=2),nn.BatchNorm2d(32))
        self.layer1p = nn.MaxPool2d((1,2))
        self.layer2a = nn.Sequential(nn.Conv2d(16,64,kernel_size=5,padding=2),nn.BatchNorm2d(64))
        self.layer2b = nn.Sequential(nn.Conv2d(32,64,kernel_size=5,padding=2),nn.BatchNorm2d(64))
        self.layer2p = nn.MaxPool2d((1,2))
        self.layer3a = nn.Sequential(nn.Conv2d(32,128,kernel_size=5,padding=2),nn.BatchNorm2d(128))
        self.layer3b = nn.Sequential(nn.Conv2d(64,128,kernel_size=5,padding=2),nn.BatchNorm2d(128))
        self.layer3p = nn.MaxPool2d((1,2))
        self.layer4a = nn.Sequential(nn.Conv2d(64,128,kernel_size=5,padding=2),nn.BatchNorm2d(128))
        self.layer4b = nn.Sequential(nn.Conv2d(64,128,kernel_size=5,padding=2),nn.BatchNorm2d(128))
        self.layer4p = nn.MaxPool2d((1,2)) #
        self.layer5a = nn.Sequential(nn.Conv2d(64,128,kernel_size=5,padding=2),nn.BatchNorm2d(128))
        self.layer5b = nn.Sequential(nn.Conv2d(64,128,kernel_size=5,padding=2),nn.BatchNorm2d(128))
        self.layer5p = nn.MaxPool2d((1,2)) #
        self.layer6 = nn.Sequential(nn.Conv2d(64,256,kernel_size=5,padding=2),nn.BatchNorm2d(256),nn.ReLU())
        self.layer6p = nn.MaxPool2d((1,4)) #
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self,x):
        y = self.layer1a(x)
        y = y[:,:16,:,:] * self.sigmoid(y[:,16:,:,:])
        y = self.layer1b(y)
        y = y[:,:16,:,:] * self.sigmoid(y[:,16:,:,:])
        y = self.layer1p(y)

        y = self.layer2a(y)
        y = y[:,:32,:,:] * self.sigmoid(y[:,32:,:,:])
        y = self.layer2b(y)
        y = y[:,:32,:,:] * self.sigmoid(y[:,32:,:,:])
        y = self.layer2p(y)

        y = self.layer3a(y)
        y = y[:,:64,:,:] * self.sigmoid(y[:,64:,:,:])
        y = self.layer3b(y)
        y = y[:,:64,:,:] * self.sigmoid(y[:,64:,:,:])
        y = self.layer3p(y)

        y = self.layer4a(y)
        y = y[:,:64,:,:] * torch.sigmoid(y[:,64:,:,:])
        y = self.layer4b(y)
        y = y[:,:64,:,:] * torch.sigmoid(y[:,64:,:,:])
        y = self.layer4p(y)

        y = self.layer5a(y)
        y = y[:,:64,:,:] * torch.sigmoid(y[:,64:,:,:])
        y = self.layer5b(y)
        y = y[:,:64,:,:] * torch.sigmoid(y[:,64:,:,:])
        y = self.layer5p(y)

        y = self.layer6(y)
        y = self.layer6p(y)

        out = y.squeeze(3).transpose(1,2)
        return out
    
    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

        
class ATclassifier(nn.Module):
    def __init__(self,nclass):
        super(ATclassifier,self).__init__()
        self.nclass = nclass
        self.cla = nn.Sequential(nn.Linear(256,nclass), nn.Sigmoid())
        self.att = nn.Sequential(nn.Linear(256, nclass + 5), nn.Softmax(dim=-1))
        #self.att = nn.Sequential(nn.Linear(256,nclass), nn.Softmax(dim=3))
        
    def forward(self, x, output_sequence=False):
        
        orig_size = x.size()
        x = x.contiguous()
        out1 = self.cla(x.view(x.size(0)*x.size(1), x.size(2)))
        out2 = self.att(x.view(x.size(0)*x.size(1), x.size(2)))
        
        out2 = out2[:, :self.nclass]
        out2 = torch.clamp(out2, 1e-7, 1)
        out1 = out1.view(orig_size[0], orig_size[1], -1)
        out2 = out2.view(orig_size[0], orig_size[1], -1)
        
        out2_target = out2[:,:,:self.nclass]
        out = torch.sum(out1 * out2_target, dim=1) / torch.sum(out2_target, dim=1)
        
        if output_sequence:
            return out1
        else:
            return out

    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

class RnnATclassifier(nn.Module):
    def __init__(self,nclass):
        super(RnnATclassifier,self).__init__()
        self.nclass = nclass
        self.rnn = nn.GRU(256, 256, 3, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(256)
    
        self.cla = nn.Sequential(nn.Linear(512,nclass), nn.Sigmoid())
        self.att = nn.Sequential(nn.Linear(512, nclass+5), nn.Softmax(dim=-1))
        #self.att = nn.Sequential(nn.Linear(256,nclass), nn.Softmax(dim=3))
        
    def forward(self, x, mask=None, output_sequence=False, output_attention=False):
        h0 = torch.zeros((6, x.size(0), 256), requires_grad=True).cuda()
        rnn_out, _ = self.rnn(x, h0)
        #rnn_out = self.bn(rnn_out)
        rnn_out = torch.tanh(rnn_out)
        orig_size = rnn_out.size()
        rnn_out = rnn_out.contiguous()
        
        out1 = self.cla(rnn_out.view(rnn_out.size(0)*rnn_out.size(1), rnn_out.size(2)))
        out2 = self.att(rnn_out.view(rnn_out.size(0)*rnn_out.size(1), rnn_out.size(2)))
        out2 = torch.clamp(out2, 1e-7, 1)
        out1 = out1.view(orig_size[0], orig_size[1], -1)
        out2 = out2.view(orig_size[0], orig_size[1], -1)
        out2_target = out2[:,:,:self.nclass]
        
        if mask is not None:
            out1 = out1 * mask
            out2_target = out2_target*mask
            
        out = torch.sum(out1 * out2_target, dim=1) / torch.sum(out2_target, dim=1)

        if output_sequence:
            if output_attention:
                return out1, out2[:,:,:self.nclass]
            else:
                return out1#*out2[:,:,:self.nclass]
        else:
            return out

    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)


        
class LinearClassifier(nn.Module):
    def __init__(self,nclass):
        super(LinearClassifier,self).__init__()
        self.nclass = nclass
        self.fc = nn.Sequential(nn.Linear(256,nclass), nn.Sigmoid())
        
    def forward(self, x, output_sequence=False):
        orig_size = x.size()
        x = x.contiguous()
        out = self.fc(x.view(x.size(0)*x.size(1), x.size(2)))
        out = out.view(orig_size[0], orig_size[1], -1)

        if output_sequence:
            return out
        else:
            return out.mean(dim=1)


    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

class RnnClassifier(nn.Module):
    def __init__(self,nclass):
        super(RnnClassifier,self).__init__()
        self.nclass = nclass
        self.rnn = nn.GRU(256, 256, 3, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(512,nclass), nn.Sigmoid())
        
    def forward(self, x, mask=None, output_sequence=True):
        #h0 = torch.zeros((6, x.size(0), 256), requires_grad=True).cuda()
        #rnn_out, _ = self.rnn(x, h0)
        rnn_out, _ = self.rnn(x)        
        orig_size = rnn_out.size()
        rnn_out = rnn_out.contiguous()
        out = self.fc(rnn_out.view(rnn_out.size(0)*rnn_out.size(1), rnn_out.size(2)))
        out = out.view(orig_size[0], orig_size[1], -1)
        
        return out


    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

        
class CascadeNet(nn.Module):
    def __init__(self,nclass):
        super(CascadeNet,self).__init__()
        self.emb_net = EmbNet()
        self.cls_net = RnnATclassifier(nclass)
        
    def forward(self, x) :
        out = self.emb_net(x)
        out = self.cls_net(out)
        return out

    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

        
