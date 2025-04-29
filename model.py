import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import copy
import numpy as np
import math
from torch_ema import ExponentialMovingAverage
import json
import matplotlib.pyplot as plt

from tqdm import tqdm
from modeling.blocks import MaskMambaBlock, MaskMambaBlock_DBM
from sklearn.metrics import confusion_matrix
from eval import segment_bars_with_confidence
from zclip import ZClip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, i, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
    
    def _sliding_window_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        
        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl 
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) 
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask 
        
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

class AttModule_mamba(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, drop_path_rate=0.3):
        super(AttModule_mamba, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = MaskMambaBlock(in_channels, drop_path_rate=drop_path_rate) # dilation
        # self.att_layer = MaskMambaBlock_DBM(in_channels, drop_path_rate=drop_path_rate) # dilation
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        m_batchsize, c1, L = x.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), padding_mask) + out
        # out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]



class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha, mamba=False, drop_path_rate=0.3):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        if not mamba:
            self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
             range(num_layers)])
        else:
            self.layers = nn.ModuleList(
                [AttModule_mamba(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha, drop_path_rate=drop_path_rate) for i in # 2**i
             range(num_layers)]
            )
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha, mamba=False, drop_path_rate=0.3):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        if not mamba:
            self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        else:
            self.layers = nn.ModuleList(
            [AttModule_mamba(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, drop_path_rate=drop_path_rate) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
    
class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, drop_path_rate=0.3, encoder_only=False):
        super(MyTransformer, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att', alpha=1)
        if encoder_only:
            self.decoders = nn.ModuleList([copy.deepcopy(Encoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, channel_masking_rate, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
        else:
            self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
        
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            if not self.encoder_only:
                out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            else:
                out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
 
        return outputs
    
from torch.nn.utils import weight_norm
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res
    

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0)  # (1, D, T)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]  # (B, D, T)
    
class SinusoidalPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=300):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

from torchinfo import summary
class MaTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, drop_path_rate=0.3):
        super(MaTransformer, self).__init__()

        self.input_dim = input_dim
        # self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, 300)) # 60s
        self.pos_embedding = SinusoidalPositionalEncoding(input_dim, max_len=300)
    
        gru_hidden_dim = input_dim * 2
        self.gru = nn.GRU(input_dim, gru_hidden_dim, batch_first=True, bidirectional=False)
        self.gru_proj = nn.Conv1d(gru_hidden_dim, input_dim * 2, 1) 

        input_dim = input_dim * 2
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att', alpha=1, mamba=True, drop_path_rate=drop_path_rate)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s), drop_path_rate=drop_path_rate)) for s in range(num_decoders)]) # num_decoders

    def forward(self, x, mask):
    

        # B, D, T = x.shape
        # x = torch.from_numpy(np.random.randn(1, self.input_dim, 300)).to(device).float()
        # mask = torch.from_numpy(np.random.randn(1, 1, 300)).to(device).float()

        # Add positional embedding
        # if T > self.pos_embedding.shape[2]:
        #     raise ValueError(f"Input length T={T} exceeds max_len={self.pos_embedding.shape[2]} in positional embedding")
        
        # x = x + self.pos_embedding[:, :, :T]

        # add pose embedding
        x = self.pos_embedding(x)  # (B, D, T)


        x = x.permute(0, 2, 1) 
        x, _ = self.gru(x)  
        x = x.permute(0, 2, 1) 
        x = self.gru_proj(x) 

        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
 
        return outputs
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=-100, reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)         
        pt = torch.exp(logpt)                     
        loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss 
    

def temporal_label_smoothing(batch_target, num_classes, window=60):
    """
    Applies temporal smoothing to one-hot encoded target labels.

    Args:
        batch_target (Tensor): shape (B, T)
        num_classes (int): number of gesture classes
        window (int): size of the smoothing kernel

    Returns:
        Tensor: shape (B, T, C)
    """
    B, T = batch_target.shape

    one_hot = F.one_hot(batch_target, num_classes=num_classes).float()  # (B, T, C)
    one_hot = one_hot.permute(0, 2, 1)  # (B, C, T)

    kernel = torch.ones(num_classes, 1, window, device=one_hot.device) / window

    smoothed = F.conv1d(one_hot, kernel, padding=window // 2, groups=num_classes)  # (B, C, T')

    # epsilon = 0.1  # Smoothing factor
    # smoothed_labels = (1 - epsilon) * one_hot + epsilon / num_classes

    # Trim or pad to match original T
    if smoothed.shape[2] > T:
        smoothed = smoothed[:, :, :T]
    elif smoothed.shape[2] < T:
        pad_amount = T - smoothed.shape[2]
        smoothed = F.pad(smoothed, (0, pad_amount))

    smoothed = smoothed / smoothed.sum(dim=1, keepdim=True).clamp(min=1e-6)
    smoothed = smoothed.permute(0, 2, 1)  # (B, T, C)

    return smoothed


def compute_confusion_matrix(predicted, target, num_classes):
    """
    predicted: Tensor of shape (B, T)
    target: Tensor of shape (B, T)
    num_classes: total number of classes
    """

    # Flatten everything
    pred_flat = predicted.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    # Remove ignored indices (like -100)
    valid_mask = target_flat != -100
    pred_flat = pred_flat[valid_mask]
    target_flat = target_flat[valid_mask]

    # Compute confusion matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=np.arange(num_classes))

    return cm


import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)  # adjust this based on your log viewer
pd.set_option("display.colheader_justify", 'center')


def print_cm(cm, class_names):
    df = pd.DataFrame(cm, index=[f"GT {i}" for i in class_names],
                         columns=[f"Pred {i}" for i in class_names])
    
    print("Confusion Matrix:")
    print(df)


class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, mamba=True, drop_path_rate=0.3, args=None):

        
        if not mamba:
            self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, encoder_only=False)
        else:
            self.model = MaTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, drop_path_rate=drop_path_rate)

        # self.model.load_state_dict(torch.load('models/Ours/Overfits/model_6.model', map_location=device), strict=False)
        self.model.load_state_dict(torch.load('models/Ours/split_1/final.model', map_location=device), strict=False) # 120
        # self.model.load_state_dict(torch.load('models/Ours/split_3/epoch-1000.model', map_location=device), strict=False) # 120

        self.model.eval()
        self.model.to(device)

        model_summary = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                input_shape = param.shape
                output_shape = param.shape
                model_summary.append([name, input_shape, output_shape, num_params])
        model_summary_df = pd.DataFrame(model_summary, columns=['Name', 'Input Shape', 'Output Shape', 'Num Parameters'])
        model_summary_df.to_csv('model_summary.csv', index=False)



        self.zclip = ZClip(
            mode="zscore",
            alpha=0.97,
            z_thresh=2.5,
            clip_option="adaptive_scaling",
            max_grad_norm=5.0,
            clip_factor=1.0
        )


        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, batch_gen_tst=None):
        self.model.train()
        self.model.to(device)

        for p in self.model.encoder.parameters():
            p.requires_grad = True

        for p in self.model.gru.parameters():
            p.requires_grad = True

        for p in self.model.gru_proj.parameters():
            p.requires_grad = True

        # Selectively unfreeze
        for p in self.model.encoder.layers[-1].parameters():
            p.requires_grad = True  # fine-tune top encoder layer
            

        n = len(self.model.encoder.layers)
        for i in range(n-1):
            for p in self.model.encoder.layers[i].parameters():
                p.requires_grad = True

        for p in self.model.decoders.parameters():
            p.requires_grad = True  # allow decoders to adapt

        def add_noise_to_model(model, std=1e-3):
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.add_(torch.randn_like(param) * std)

        add_noise_to_model(self.model, std=1e-3)

        # self.model.pos_embedding.requires_grad = False  # slow update if needed
        # set to zero
        # self.model.pos_embedding.data.zero_()

        base_lr = learning_rate
        optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': base_lr * 0.1},
            {'params': self.model.gru.parameters(), 'lr': base_lr * 0.1},
            {'params': self.model.decoders.parameters(), 'lr': base_lr},
            {'params': self.model.gru_proj.parameters(), 'lr': base_lr * 0.1},
            # {'params': self.model.pos_embedding, 'lr': base_lr * 0.05},
        ])
        ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)

        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        print('LR:{}'.format(learning_rate))
        
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)

        history_loss = []
        history_acc = []
        

        for epoch in range(num_epochs):
            epoch_loss = 0
        
            if epoch % 10 == 0:
                class_wise_corr = [0] * self.num_classes
                class_wise_total = [0] * self.num_classes
                class_wise_predicted = [0] * self.num_classes
                all_predictions = []
                all_targets = []
                total = 0
                correct = 0

            total_num = len(batch_gen.list_of_examples)
            bar = tqdm(total=total_num, desc='Training', unit='batch')

            accum_steps = 32
            optimizer.zero_grad()
            step_count = 0

            loss_max = 0
            total_class_loss = 0
            total_smooth_loss = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size, False)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)

                # smooth the target labels with a window of 30 frames (0.5s)
                # smooth_target = torch.zeros(batch_target.shape[0], batch_target.shape[1], self.num_classes, device=device)
                smooth_target = temporal_label_smoothing(batch_target, self.num_classes, window=60)  # (B, T, C)

                optimizer.zero_grad()
                ps = self.model(batch_input, mask)
                
                # weights = torch.zeros(self.num_classes, device=device)
                # for i in range(self.num_classes):
                #     weights[i] = (batch_target == i).float().sum()
            
                # non_zero_weights_idx = torch.where(weights != 0)[0]
                # weights[non_zero_weights_idx] = 1 / weights[non_zero_weights_idx]
                # weights[non_zero_weights_idx] = weights[non_zero_weights_idx] / torch.sum(weights[non_zero_weights_idx]) * self.num_classes
                

                weights = torch.tensor([0.15803394, 1.15718843, 1.18161826, 1.13808255, 1.16453608, 1.20054073], dtype=torch.float32, device=device)
               
                # self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
                # self.ce = FocalLoss(gamma=2, weight=weights, reduction='mean')
                
                loss = 0
                for p in ps:
                    # if epoch > 100:
                    smooth_loss = self.mse(
                        F.log_softmax(p[:, :, 1:], dim=1),
                        F.log_softmax(p[:, :, :-1], dim=1)
                    )
                    smooth_loss = torch.clamp(smooth_loss, min=0, max=16)
                    smooth_loss = torch.mean(smooth_loss) * min(0.15, epoch / 100)
                    loss += smooth_loss

                    assert p.shape[2] == batch_target.shape[1], f"Mismatch: pred_len={p.shape[2]}, target_len={batch_target.shape[1]}"

                    class_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1)) / accum_steps
                    loss += class_loss
                    total_class_loss += class_loss.item()
                    total_smooth_loss += smooth_loss.item()


                epoch_loss += loss.item()
                
                (loss / accum_steps).backward()

                step_count += 1

                conf, predicted = torch.max(ps.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                all_predictions.append(predicted)
                all_targets.append(batch_target)

                if loss.item() > loss_max:
                    loss_max = loss.item()
                    data = batch_input, batch_target, mask, vids, conf, predicted

                
                if step_count % accum_steps == 0 or step_count == total_num:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                    if step_count == total_num:
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        print(f"[Before] Grad Norm: {total_norm:.2f} | Class counts: {torch.bincount(batch_target.view(-1))}")

                    
                    self.zclip.step(self.model)
                    optimizer.step()
                    # ema.update()
            
                if step_count == total_num:
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"[After] Grad Norm: {total_norm:.2f} | Class counts: {torch.bincount(batch_target.view(-1))}")


                if step_count % accum_steps == 0 or step_count == total_num:
                    optimizer.zero_grad()


                for i in range(self.num_classes):
                    class_wise_corr[i] += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1) * (batch_target == i).float()).sum().item()
                    class_wise_total[i] += ((mask[:, 0, :].squeeze(1) * (batch_target == i).float()).sum().item())
                    class_wise_predicted[i] += ((mask[:, 0, :].squeeze(1) * (predicted == i).float()).sum().item())

                bar.update(batch_size)
                bar.set_postfix(loss=loss.item() / accum_steps, acc=correct / total)
                bar.refresh()
            bar.close()


            # visualie the data with the max loss
            # read env.json, if deubg true;
            debug = json.load(open('env.json', 'r'))['debug']
            if debug:
                import matplotlib.pyplot as plt
                from matplotlib import animation
                # Create two subplots: left for scatter animation and right for signal plot animation
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Add a text object to display the target label on the scatter plot
                target_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes)
                
                # Get the original feature vector of shape (C, L) (C: number of channels, L: number of frames)
                orig_feature = data[0][0].detach().cpu().numpy()  # shape: (C, L)
                C, L = orig_feature.shape
                
                # For the signal plot, display the actual feature vector (flattened) of each frame.
                # Initially, plot the first frame's feature vector.
                line, = ax2.plot(np.arange(C), orig_feature[:, 0], color='b')
                ax2.set_xlim([0, C])
                ax2.set_ylim([orig_feature.min() - 0.1 * abs(orig_feature.min()),
                            orig_feature.max() + 0.1 * abs(orig_feature.max())])
                ax2.set_title("Feature Signal (Flattened)")
                ax2.set_xlabel("Feature Index")
                ax2.set_ylabel("Activation")
                
                # Prepare scatter data by reshaping the original feature vector.
                # We assume the channel dimension is even; each pair corresponds to (x, y) for plotting.
                reshaped_feature = orig_feature.reshape((C // 2, 2, L))
                
                # Define the update function for the animation: update both scatter and signal line.
                def update(frame, reshaped_feature, scatter, target, target_text, line, orig_feature):
                    # Update scatter plot in ax1 using the reshaped feature vector for the current frame
                    offsets = np.column_stack((reshaped_feature[:, 0, frame], reshaped_feature[:, 1, frame]))
                    scatter.set_offsets(offsets)
                    target_text.set_text("Target: {}".format(target[frame]))

                    # Update signal plot in ax2 with the flattened feature vector of the current frame
                    line.set_data(np.arange(C),  np.vstack((reshaped_feature[:, 0, frame], reshaped_feature[:, 1, frame])))
                    return scatter, target_text, line
                
                # Get target for animation (assumed to be a 1D array of length L)
                target = data[1][0].detach().cpu().numpy()
                
                # Initialize scatter plot on ax1 using the initial frame of reshaped_feature
                colors = np.linspace(0, 1, reshaped_feature.shape[0])
                cmap = plt.get_cmap('RdYlGn')
                scatter = ax1.scatter(reshaped_feature[:, 0, 0], reshaped_feature[:, 1, 0],
                                    c=colors, cmap=cmap, marker='o')
                ax1.set_title("Scatter Animation")
                ax1.set_xlim([-1, 1])
                ax1.set_ylim([-1, 1])
                
                # Create the animation: update both subplots for each frame
                ani = animation.FuncAnimation(
                    fig,
                    update,
                    frames=L,
                    fargs=(reshaped_feature, scatter, target, target_text, line, orig_feature),
                    interval=100,
                    repeat=False
                )
                
                plt.show()

            # Compute average loss and accuracy for the epoch
            epoch_loss_avg = epoch_loss / len(batch_gen.list_of_examples)
            class_loss = total_class_loss / len(batch_gen.list_of_examples)
            smooth_loss = total_smooth_loss / len(batch_gen.list_of_examples)

            epoch_acc = float(correct) / total
            history_loss.append(epoch_loss_avg)
            history_acc.append(epoch_acc)
            
            if (epoch + 1) % 100 == 0 and batch_gen_tst is not None:
                for i in range(len(ps)):
                    confidence, predicted = torch.max(F.softmax(ps[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
                    segment_bars_with_confidence('tmp/{}_stage{}.png'.format(epoch, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

            scheduler.step(epoch_loss)
            batch_gen.reset()
            current_lr = optimizer.param_groups[0]['lr']

            print("[epoch %d]: epoch loss = %f, ce loss = %f, smooth loss = %f,  acc = %f, learning_rate = %f" % (epoch + 1, epoch_loss_avg, class_loss, smooth_loss, epoch_acc, current_lr))
            accs = [float(class_wise_corr[i]) / class_wise_predicted[i] if class_wise_predicted[i] != 0 else 0 for i in range(self.num_classes)]
            recalls = [float(class_wise_corr[i]) / class_wise_total[i] if class_wise_total[i] != 0 else 0 for i in range(self.num_classes)]
            F1s = [2 * accs[i] * recalls[i] / (accs[i] + recalls[i]) if accs[i] + recalls[i] != 0 else 0 for i in range(self.num_classes)]

            cm = compute_confusion_matrix(torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0), num_classes=self.num_classes)
            print_cm(cm, class_names=['no_action', 'waving', 'point', 'bigger', 'smaller', 'thumbs_up'])


            print("\n{:^7} | {:^6} | {:^6} | {:^6}".format("Class", "Acc", "Recall", "F1"))
            print("-" * 31)
            for i in range(self.num_classes):
                print("{:^7} | {:^6.2f} | {:^6.2f} | {:^6.2f}".format(i, accs[i], recalls[i], F1s[i]))

            print(f'Total class 3: {class_wise_corr[3]} and total class 4: {class_wise_corr[4]}')
            print(f'Total class 3: {class_wise_total[3]} and total class 4: {class_wise_total[4]}')
            print("--------------------------------------------------")
            
            if (epoch + 1) % 1000 == 0 and batch_gen_tst is not None:
                self.test(batch_gen_tst, epoch)
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
        
        # Plot and save loss history
            if (epoch + 1)% 1000 == 0:
                epochs = range(1, len(history_loss) + 1)
                fig, ax1 = plt.subplots()

                color = 'tab:red'
                ax1.set_xlabel('Epochs')
                ax1.set_ylabel('Loss', color=color)
                ax1.plot(epochs, history_loss, color=color, label='Training Loss')
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Accuracy', color=color)
                ax2.plot(epochs, history_acc, color=color, label='Training Accuracy')
                ax2.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()
                plt.title('Training Loss and Accuracy History')
                plt.savefig(save_dir + f'/training_loss_accuracy-history_epoch_{epoch + 1}.png')
                plt.close()

    def test(self, batch_gen_tst, epoch):
        self.model.eval()
        correct = 0
        total = 0
        if_warp = False  # When testing, always false

        class_wise_corr = [0] * self.num_classes
        class_wise_total = [0] * self.num_classes
        class_wise_predicted = [0] * self.num_classes

        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1, if_warp)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                p = self.model(batch_input, mask)
                _, predicted = torch.max(p.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                for i in range(self.num_classes):
                    class_wise_corr[i] += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1) * (batch_target == i).float()).sum().item()
                    class_wise_total[i] += ((mask[:, 0, :].squeeze(1) * (batch_target == i).float()).sum().item())
                    class_wise_predicted[i] += ((mask[:, 0, :].squeeze(1) * (predicted == i).float()).sum().item())

        accs = [float(class_wise_corr[i]) / class_wise_predicted[i] if class_wise_predicted[i] != 0 else 0 for i in range(self.num_classes)]
        print("Class wise accs: ", [round(acc, 2) for acc in accs])
        
        recalls = [float(class_wise_corr[i]) / class_wise_total[i] if class_wise_total[i] != 0 else 0 for i in range(self.num_classes)]
        print("Class wise recall: ", [round(rec, 2) for rec in recalls])

        acc = float(correct) / total if total != 0 else 0
        
        print("Class wise F1s: ", [round(2 * accs[i] * recalls[i] / (accs[i] + recalls[i]), 2) if accs[i] + recalls[i] != 0 else 0 for i in range(self.num_classes)])


        print("---[epoch %d]---: tst acc = %f" % (epoch + 1, acc))
        print("--------------------------------------------------")

        self.model.train()
        batch_gen_tst.reset()

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))

            batch_gen_tst.reset()
            import time
            
            time_start = time.time()
            while batch_gen_tst.has_next():
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0]
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            time_end = time.time()
            
            

if __name__ == '__main__':
    pass
