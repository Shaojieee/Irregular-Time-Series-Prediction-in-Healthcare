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
        
        self.Wq = nn.Parameter(torch.empty(self.N, self.h, d, self.dk), requires_grad=True)
        initialise_parameters(self.Wq, 'glorot_uniform')
        
        self.Wk = nn.Parameter(torch.empty(self.N, self.h, d, self.dk), requires_grad=True)
        initialise_parameters(self.Wk, 'glorot_uniform')
        
        self.Wv = nn.Parameter(torch.empty(self.N, self.h, d, self.dk), requires_grad=True)
        initialise_parameters(self.Wv, 'glorot_uniform')
        
        self.Wo = nn.Parameter(torch.empty(self.N, self.dv*self.h, d), requires_grad=True)
        initialise_parameters(self.Wo, 'glorot_uniform')
        
        
        self.W1 = nn.Parameter(torch.empty(self.N, d, self.dff), requires_grad=True)
        initialise_parameters(self.W1, 'glorot_uniform')
        
        self.b1 = nn.Parameter(torch.empty(self.N, self.dff), requires_grad=True)
        initialise_parameters(self.b1, 'zeros')
        
        self.W2 = nn.Parameter(torch.empty(self.N, self.dff, d), requires_grad=True)
        initialise_parameters(self.W2, 'glorot_uniform')
        
        self.b2 = nn.Parameter(torch.empty(self.N, d), requires_grad=True)
        initialise_parameters(self.b2, 'zeros')
        
        
        self.gamma = nn.Parameter(torch.empty(2*self.N,), requires_grad=True)
        initialise_parameters(self.gamma, 'ones')
        
        self.beta = nn.Parameter(torch.empty(2*self.N,), requires_grad=True)
        initialise_parameters(self.beta, 'zeros')
        
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.identity = nn.Identity()
        
    def forward(self, X, mask, mask_value=-1e-30):
        mask = torch.unsqueeze(mask, dim=-2)
        
        # N times of transformer
        for i in range(self.N):
            mha_ops = []
            print(f'Transformer {i}')
            # h heads for multi headed attention
            for j in range(self.h):
                print(f'Head {j}')
                q = torch.matmul(X, self.Wq[i,j,:,:])
                k = torch.matmul(X, self.Wk[i,j,:,:]).permute(0,2,1)
                v = torch.matmul(X, self.Wv[i,j,:,:])
                print(f'X:{X.shape}, w:{self.Wq[i,j,:,:].shape}, q: {q.shape}, k: {k.shape}, v: {v.shape}')
                A = torch.bmm(q, k)
                print(f'A: {A.shape}')
                A = mask * A + (1-mask) * mask_value
                
                def dropped_A():
                    dp_mask = (torch.rand_like(A)>=self.dropout).type(dtype=torch.float32)
                    return A*dp_mask + (1-dp_mask)*mask_value
                # Dropout
                if self.training:
                    print('In dropout')
                    A = dropped_A()
                else:
                    A = self.identity(A)
                    
                A = nn.functional.softmax(A, dim=-1)
                
                mha_ops.append(torch.bmm(A, v))
                print(f'mha: {mha_ops[j].shape}')
            
            conc = torch.cat(mha_ops, dim=-1)
            print(f'conc: {conc.shape}')
            proj = torch.matmul(conc, self.Wo[i,:,:])
            print(f'proj: {proj.shape}')
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


class BertForRepresentation(nn.Module):

    def __init__(self, model, model_name):
        super().__init__()

        self.model = model
        self.dropout = torch.nn.Dropout(model.config.hidden_dropout_prob)
        self.model_name = model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None , doc_idx_list=None):
        txt_arr = []

        for input_ids,attention_mask in zip(input_ids_sequence,attention_mask_sequence):

            if 'Longformer' in self.model_name:

                attention_mask-=1
                text_embeddings=self.model(input_ids, global_attention_mask=attention_mask)

            else:
                text_embeddings=self.model(input_ids, attention_mask=attention_mask)

            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr=torch.stack(txt_arr)
        return txt_arr


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
        
        self.Wq = nn.Parameter(torch.empty(self.h, d, self.dk), requires_grad=True)
        initialise_parameters(self.Wq, 'glorot_uniform')
        
        self.Wk = nn.Parameter(torch.empty(self.h, d, self.dk), requires_grad=True)
        initialise_parameters(self.Wk, 'glorot_uniform')
        
        self.Wv = nn.Parameter(torch.empty(self.h, d, self.dk), requires_grad=True)
        initialise_parameters(self.Wv, 'glorot_uniform')
        
        self.Wo = nn.Parameter(torch.empty(self.dv*self.h, d), requires_grad=True)
        initialise_parameters(self.Wo, 'glorot_uniform')
        
        
        self.W1 = nn.Parameter(torch.empty(d, self.dff), requires_grad=True)
        initialise_parameters(self.W1, 'glorot_uniform')
        
        self.b1 = nn.Parameter(torch.empty(self.dff), requires_grad=True)
        initialise_parameters(self.b1, 'zeros')
        
        self.W2 = nn.Parameter(torch.empty(self.dff, d), requires_grad=True)
        initialise_parameters(self.W2, 'glorot_uniform')
        
        self.b2 = nn.Parameter(torch.empty(d), requires_grad=True)
        initialise_parameters(self.b2, 'zeros')
        
        
        self.gamma = nn.Parameter(torch.empty(2,), requires_grad=True)
        initialise_parameters(self.gamma, 'ones')
        
        self.beta = nn.Parameter(torch.empty(2,), requires_grad=True)
        initialise_parameters(self.beta, 'zeros')
        
        
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.identity = nn.Identity()
    
    def forward(self, query, key, value, mask, mask_value=-1e-30):
        mask = torch.unsqueeze(mask,dim=-2)

        mha_ops = []

        for j in range(self.h):
            print(f'Head {j}')
            q = torch.matmul(query, self.Wq[j,:,:])
            k = torch.matmul(key, self.Wk[j,:,:]).permute(0,2,1)
            v = torch.matmul(value, self.Wv[j,:,:])
            print(f'w:{self.Wq[j,:,:].shape}, q: {q.shape}, k: {k.shape}, v: {v.shape}')
            A = torch.bmm(q, k)
            print(f'A: {A.shape}')
            A = mask * A + (1-mask) * mask_value
            
            def dropped_A():
                dp_mask = (torch.rand_like(A)>=self.dropout).type(dtype=torch.float32)
                return A*dp_mask + (1-dp_mask)*mask_value
            # Dropout
            if self.training:
                print('In dropout')
                A = dropped_A()
            else:
                A = self.identity(A)
                
            A = nn.functional.softmax(A, dim=-1)
            
            mha_ops.append(torch.bmm(A, v))
            print(f'mha: {mha_ops[j].shape}')
        
        conc = torch.cat(mha_ops, dim=-1)
        print(f'conc: {conc.shape}')
        proj = torch.matmul(conc, self.Wo)
        print(f'proj: {proj.shape}')
        # Dropout
        if self.training:
            proj = self.identity(self.dropout_layer(proj))
        else:
            proj = self.identity(proj)
        
        value = proj
        # Layer Normalisation
        # mean = torch.mean(value, dim=-1, keepdim=True)
        # variance = torch.mean(torch.square(value - mean), axis=-1 ,keepdims=True)
        # std = torch.sqrt(variance + self.epsilon)
        # value  = (value-mean)/std
        # value = value * self.gamma[0] + self.beta[0]
        
        # # FFN
        # ffn_op = torch.add(torch.matmul(nn.functional.relu(torch.add(torch.matmul(value, self.W1), self.b1)), self.W2),self.b2)
        # # FFN Dropout
        # if self.training:
        #     ffn_op = self.dropout_layer(ffn_op)
        # else:
        #     ffn_op = self.identity(ffn_op)
        
        # # Add
        # value = value + ffn_op
        # # Layer Normalisation
        # mean = torch.mean(value, dim=-1, keepdim=True)
        # variance = torch.mean(torch.square(value - mean), axis=-1 ,keepdims=True)
        # std = torch.sqrt(variance + self.epsilon)
        # value = (value-mean)/std
        # value = value*self.gamma[1] + self.beta[1]

        return value


class MultiHeadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim), requires_grad=True)
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim), requires_grad=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim), requires_grad=True)
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim), requires_grad=True)
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        # import pdb;
        # pdb.set_trace()
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, q_seq_len=None, kv_seq_len=None,):
        super().__init__()

        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.q_seq_len=q_seq_len
        self.kv_seq_len=kv_seq_len

        if self.q_seq_len!=None:
            self.embed_positions_q=nn.Embedding(self.q_seq_len,embed_dim,padding_idx=0)
            nn.init.normal_(self.embed_positions_q.weight, std=0.02)

        if self.kv_seq_len!=None:
            self.embed_positions_kv=nn.Embedding(self.kv_seq_len,embed_dim)
            nn.init.normal_(self.embed_positions_kv.weight, std=0.02)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """


        x=x_in
        length_x = x.size(0) # (length,Batch_size,input_dim)
        x = self.embed_scale * x_in
        if self.q_seq_len is not None:
            position_x = torch.tensor(torch.arange(length_x),dtype=torch.long)
            x += (self.embed_positions_q(position_x).unsqueeze(0)).transpose(0,1)  # Add positional embedding
        x =F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions

            length_kv = x_in_k.size(0) # (Batch_size,length,input_dim)
            position_kv = torch.tensor(torch.arange(length_kv),dtype=torch.long)

            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.kv_seq_len is not None:
                x_k += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
                x_v += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)


        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = nn.Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:bpbpp
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]