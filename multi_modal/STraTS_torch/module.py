import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import math


def initialise_linear_layer(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
  
            
def initialise_parameters(layer, method):
    if isinstance(layer, nn.Parameter):
        if method=='glorot_uniform':
            torch.nn.init.xavier_uniform_(layer)
        elif method=='zeros':
            torch.nn.init.zeros_(layer)
        elif method=='ones':
            torch.nn.init.ones_(layer)


class CVE(nn.Module):
    def __init__(self, hid_dim, output_dim):
        super(CVE, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=1, out_features=hid_dim, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=output_dim, bias=False)
        )
        self.stack.apply(initialise_linear_layer)
        
    def forward(self, X):
        X = X.unsqueeze(dim=-1)
        return self.stack(X)


class TVE(nn.Module):
    def __init__(self, hid_dim, output_dim, input_dim=768):
        super(TVE, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hid_dim, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=output_dim, bias=False)
        )
        self.stack.apply(initialise_linear_layer)
        
    def forward(self, X):
        # X = X.unsqueeze(dim=-1)
        return self.stack(X)


class Time2Vec(nn.Module):
    def __init__(self, output_dim):
        super(Time2Vec, self).__init__()

        self.periodic = nn.Linear(1, output_dim-1)
        self.linear = nn.Linear(1, 1)

    def forward(self, time):
        time = time.unsqueeze(-1)
        periodic_out = torch.sin(self.periodic(time))
        linear_out = self.linear(time)
        return torch.cat([linear_out, periodic_out],-1)


class Attention(nn.Module):
    def __init__(self, d, hid_dim):
        super(Attention, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=d, out_features=hid_dim, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=hid_dim, out_features=1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-2)
        self.stack.apply(initialise_linear_layer)
    
    def forward(self, X, mask, mask_value=-1e9):
        attn_weights = self.stack(X)
        mask = torch.unsqueeze(mask, dim=-1)

        mask_value = -1e+30 if attn_weights.dtype == torch.float32 else -1e+4
        attn_weights = mask*attn_weights + (1-mask)*mask_value
        attn_weights = self.softmax(attn_weights)
        
        return attn_weights


class STraTS_Transformer(nn.Module):
    def __init__(self, d, N=2, h=8, dk=None, dv=None, dff=None, dropout=0, epsilon=1e-07):
        super(STraTS_Transformer, self).__init__()
        
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        self.epsilon = epsilon * epsilon
        if self.dk==None:
            self.dk = d // self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        
        self.Wq = nn.Parameter(torch.empty(self.N, self.h, d, self.dk))
        initialise_parameters(self.Wq, 'glorot_uniform')
        
        self.Wk = nn.Parameter(torch.empty(self.N, self.h, d, self.dk))
        initialise_parameters(self.Wk, 'glorot_uniform')
        
        self.Wv = nn.Parameter(torch.empty(self.N, self.h, d, self.dv))
        initialise_parameters(self.Wv, 'glorot_uniform')
        
        self.Wo = nn.Parameter(torch.empty(self.N, self.dv*self.h, d))
        initialise_parameters(self.Wo, 'glorot_uniform')
        
        
        self.W1 = nn.Parameter(torch.empty(self.N, d, self.dff))
        initialise_parameters(self.W1, 'glorot_uniform')
        
        self.b1 = nn.Parameter(torch.empty(self.N, self.dff))
        initialise_parameters(self.b1, 'zeros')
        
        self.W2 = nn.Parameter(torch.empty(self.N, self.dff, d))
        initialise_parameters(self.W2, 'glorot_uniform')
        
        self.b2 = nn.Parameter(torch.empty(self.N, d))
        initialise_parameters(self.b2, 'zeros')
        
        
        self.gamma = nn.Parameter(torch.empty(2*self.N,))
        initialise_parameters(self.gamma, 'ones')
        
        self.beta = nn.Parameter(torch.empty(2*self.N,))
        initialise_parameters(self.beta, 'zeros')
        
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.identity = nn.Identity()
        
    def forward(self, X, mask, mask_value=-1e-30):
        mask = torch.unsqueeze(mask, dim=-2)
        
        # N times of transformer
        for i in range(self.N):
            mha_ops = []
            # print(f'Transformer {i}')
            # h heads for multi headed attention
            for j in range(self.h):
                # print(f'Head {j}')
                q = torch.matmul(X, self.Wq[i,j,:,:])
                k = torch.matmul(X, self.Wk[i,j,:,:]).permute(0,2,1)
                v = torch.matmul(X, self.Wv[i,j,:,:])
                # print(f'X:{X.shape}, w:{self.Wq[i,j,:,:].shape}, q: {q.shape}, k: {k.shape}, v: {v.shape}')
                A = torch.bmm(q, k)
                # print(f'A: {A.shape}')
                A = mask * A + (1-mask) * mask_value
                
                def dropped_A():
                    dp_mask = (torch.rand_like(A)>=self.dropout).type(dtype=torch.float32)
                    return A*dp_mask + (1-dp_mask)*mask_value
                # Dropout
                if self.training:
                    # print('In dropout')
                    A = dropped_A()
                else:
                    A = self.identity(A)
                    
                A = nn.functional.softmax(A, dim=-1)
                
                mha_ops.append(torch.bmm(A, v))
                # print(f'mha: {mha_ops[j].shape}')
            
            conc = torch.cat(mha_ops, dim=-1)
            # print(f'conc: {conc.shape}')
            proj = torch.matmul(conc, self.Wo[i,:,:])
            # print(f'proj: {proj.shape}')
            # Dropout
            if self.training:
                proj = self.identity(self.dropout_layer(proj))
            else:
                proj = self.identity(proj)
            
            # Add
            X = X + proj
            # Layer Normalisation
            mean = torch.mean(X, dim=-1, keepdim=True)
            variance = torch.mean(torch.square(X - mean), axis=-1 ,keepdims=True)
            std = torch.sqrt(variance + self.epsilon)
            X  = (X-mean)/std
            X = X * self.gamma[2*i] + self.beta[2*i]
            
            # FFN
            ffn_op = torch.add(torch.matmul(nn.functional.relu(torch.add(torch.matmul(X, self.W1[i,:,:]), self.b1[i,:])), self.W2[i,:,:]),self.b2[i,:])
            # FFN Dropout
            if self.training:
                ffn_op = self.dropout_layer(ffn_op)
            else:
                ffn_op = self.identity(ffn_op)
            
            # Add
            X = X + ffn_op
            # Layer Normalisation
            mean = torch.mean(X, dim=-1, keepdim=True)
            variance = torch.mean(torch.square(X - mean), axis=-1 ,keepdims=True)
            std = torch.sqrt(variance + self.epsilon)
            X = (X-mean)/std
            X = X*self.gamma[2*i+1] + self.beta[2*i+1]
            
        return X            


class STraTS_MultiTimeAttention(nn.Module):
    def __init__(self, d, h=8, dk=None, dv=None, dff=None, dropout=0, epsilon=1e-07):
        super(STraTS_MultiTimeAttention, self).__init__()
        
        self.h, self.dk, self.dv, self.dff, self.dropout = h, dk, dv, dff, dropout
        self.epsilon = epsilon * epsilon
        if self.dk==None:
            self.dk = d // self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        
        self.Wq = nn.Parameter(torch.empty(self.h, d, self.dk))
        initialise_parameters(self.Wq, 'glorot_uniform')
        
        self.Wk = nn.Parameter(torch.empty(self.h, d, self.dk))
        initialise_parameters(self.Wk, 'glorot_uniform')
        
        self.Wv = nn.Parameter(torch.empty(self.h, d, self.dv))
        initialise_parameters(self.Wv, 'glorot_uniform')
        
        self.Wo = nn.Parameter(torch.empty(self.dv*self.h, d))
        initialise_parameters(self.Wo, 'glorot_uniform')
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.identity = nn.Identity()
    
    def forward(self, query, key, value, mask, mask_value=-1e-30):
        mask = torch.unsqueeze(mask,dim=-2)

        mha_ops = []

        for j in range(self.h):
            # print(f'Head {j}')
            q = torch.matmul(query, self.Wq[j,:,:])
            k = torch.matmul(key, self.Wk[j,:,:]).permute(0,2,1)
            v = torch.matmul(value, self.Wv[j,:,:])
            # print(f'w:{self.Wq[j,:,:].shape}, q: {q.shape}, k: {k.shape}, v: {v.shape}')
            A = torch.bmm(q, k)
            # print(f'A: {A.shape}')
            A = mask * A + (1-mask) * mask_value
            
            def dropped_A():
                dp_mask = (torch.rand_like(A)>=self.dropout).type(dtype=torch.float32)
                return A*dp_mask + (1-dp_mask)*mask_value
            # Dropout
            if self.training:
                # print('In dropout')
                A = dropped_A()
            else:
                A = self.identity(A)
                
            A = nn.functional.softmax(A, dim=-1)
            
            mha_ops.append(torch.bmm(A, v))
            # print(f'mha: {mha_ops[j].shape}')
        
        conc = torch.cat(mha_ops, dim=-1)
        # print(f'conc: {conc.shape}')
        return conc


class STraTS_SpecialTransformer(nn.Module):
    def __init__(self, d, h=8, dk=None, dv=None, dff=None, dropout=0, epsilon=1e-07):
        super(STraTS_SpecialTransformer, self).__init__()
        
        self.h, self.dk, self.dv, self.dff, self.dropout = h, dk, dv, dff, dropout
        self.epsilon = epsilon * epsilon
        if self.dk==None:
            self.dk = d // self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        
        self.Wq = nn.Parameter(torch.empty(self.h, d, self.dk))
        initialise_parameters(self.Wq, 'glorot_uniform')
        
        self.Wk = nn.Parameter(torch.empty(self.h, d, self.dk))
        initialise_parameters(self.Wk, 'glorot_uniform')
        
        self.Wv = nn.Parameter(torch.empty(self.h, d, self.dv))
        initialise_parameters(self.Wv, 'glorot_uniform')
        
        self.Wo = nn.Parameter(torch.empty(self.dv*self.h, d))
        initialise_parameters(self.Wo, 'glorot_uniform')

        self.W1 = nn.Parameter(torch.empty(d, self.dff))
        initialise_parameters(self.W1, 'glorot_uniform')
        
        self.b1 = nn.Parameter(torch.empty(self.dff))
        initialise_parameters(self.b1, 'zeros')
        
        self.W2 = nn.Parameter(torch.empty(self.dff, d))
        initialise_parameters(self.W2, 'glorot_uniform')
        
        self.b2 = nn.Parameter(torch.empty(d))
        initialise_parameters(self.b2, 'zeros')
        
        
        self.gamma = nn.Parameter(torch.empty(2,))
        initialise_parameters(self.gamma, 'ones')
        
        self.beta = nn.Parameter(torch.empty(2,))
        initialise_parameters(self.beta, 'zeros')
        
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.identity = nn.Identity()
        
    def forward(self, query, key, value, mask, mask_value=-1e-30):
        mask = torch.unsqueeze(mask,dim=-2)

        mha_ops = []

        for j in range(self.h):
            # print(f'Head {j}')
            q = torch.matmul(query, self.Wq[j,:,:])
            k = torch.matmul(key, self.Wk[j,:,:]).permute(0,2,1)
            v = torch.matmul(value, self.Wv[j,:,:])
            # print(f'w:{self.Wq[j,:,:].shape}, q: {q.shape}, k: {k.shape}, v: {v.shape}')
            A = torch.bmm(q, k)
            # print(f'A: {A.shape}')
            A = mask * A + (1-mask) * mask_value
            
            def dropped_A():
                dp_mask = (torch.rand_like(A)>=self.dropout).type(dtype=torch.float32)
                return A*dp_mask + (1-dp_mask)*mask_value
            # Dropout
            if self.training:
                # print('In dropout')
                A = dropped_A()
            else:
                A = self.identity(A)
                
            A = nn.functional.softmax(A, dim=-1)
            
            mha_ops.append(torch.bmm(A, v))
            # print(f'mha: {mha_ops[j].shape}')
        
        conc = torch.cat(mha_ops, dim=-1)
        # print(f'conc: {conc.shape}')
        proj = torch.matmul(conc, self.Wo)
        # print(f'proj: {proj.shape}')
        # Dropout
        if self.training:
            proj = self.identity(self.dropout_layer(proj))
        else:
            proj = self.identity(proj)
        
        # Add
        X = value + proj
        # Layer Normalisation
        mean = torch.mean(X, dim=-1, keepdim=True)
        variance = torch.mean(torch.square(X - mean), axis=-1 ,keepdims=True)
        std = torch.sqrt(variance + self.epsilon)
        X  = (X-mean)/std
        X = X * self.gamma[0] + self.beta[0]
        
        # FFN
        ffn_op = torch.add(torch.matmul(nn.functional.relu(torch.add(torch.matmul(X, self.W1), self.b1)), self.W2),self.b2)
        # FFN Dropout
        if self.training:
            ffn_op = self.dropout_layer(ffn_op)
        else:
            ffn_op = self.identity(ffn_op)
        
        # Add
        X = X + ffn_op
        # Layer Normalisation
        mean = torch.mean(X, dim=-1, keepdim=True)
        variance = torch.mean(torch.square(X - mean), axis=-1 ,keepdims=True)
        std = torch.sqrt(variance + self.epsilon)
        X = (X-mean)/std
        X = X*self.gamma[1] + self.beta[1]
        
        return X            


class MultiTimeAttention(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim=16,
        hid_dim=16,
        num_heads=1, 
        dropout=0.2
    ):

        super(MultiTimeAttention, self).__init__()
        assert hid_dim % num_heads == 0
        self.hid_dim = hid_dim
        self.embed_time_k = hid_dim // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.linears = nn.ModuleList([nn.Linear(hid_dim, hid_dim),
                                      nn.Linear(hid_dim, hid_dim),
                                      nn.Linear(input_dim*num_heads, output_dim)])


    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        
        # print(f' MultiTime Attention scores: {scores.shape}')
        
        scores = scores.unsqueeze(-1)
        # print(f' MultiTime Attention scores: {scores.shape}')
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)

            # print(f' MultiTime Attention mask: {mask.shape}')
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0,  -1e+30 if scores.dtype == torch.float32 else -1e+4)
        p_attn = F.softmax(scores, dim = -2)
        # print(f' MultiTime Attention p_attn: {p_attn.shape}')
        if dropout is not None:
            p_attn=F.dropout(p_attn, p=dropout, training=self.training)
            # p_attn = dropout(p_attn)
        # print(f' MultiTime Attention value.unsqueeze(-3): {value.unsqueeze(-3).shape}')
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn


    def forward(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        # import pdb; pdb.set_trace()
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        
        # print(f' MultiTime Attention query: {query.shape}')
        # print(f' MultiTime Attention key: {key.shape}')
        # print(f' MultiTime Attention value: {value.shape}')

        x, _ = self.attention(query, key, value, mask, self.dropout)


        # print(f' MultiTime Attention output: {x.shape}')
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        
        # print(f' MultiTime Attention output: {x.shape}')
        return self.linears[-1](x)