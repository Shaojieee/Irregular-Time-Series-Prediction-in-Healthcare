import torch.nn as nn
import torch
import numpy as np


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
    
    def forward(self, X, mask, mask_value=-1e30):
        attn_weights = self.stack(X)
        mask = torch.unsqueeze(mask, dim=-1)
        attn_weights = mask*attn_weights + (1-mask)*mask_value
        attn_weights = self.softmax(attn_weights)
        
        return attn_weights


class Transformer(nn.Module):
    def __init__(self, d, N=2, h=8, dk=None, dv=None, dff=None, dropout=0, epsilon=1e-07):
        super(Transformer, self).__init__()
        
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
        
        self.Wv = nn.Parameter(torch.empty(self.N, self.h, d, self.dk))
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
            # h heads for multi headed attention
            for j in range(self.h):
                q = torch.matmul(X, self.Wq[i,j,:,:])
                k = torch.matmul(X, self.Wk[i,j,:,:]).permute(0,2,1)
                v = torch.matmul(X, self.Wv[i,j,:,:])
                A = torch.bmm(q, k)
                A = mask * A + (1-mask) * mask_value
                
                def dropped_A():
                    dp_mask = (torch.rand_like(A)>=self.dropout).type(dtype=torch.float32)
                    return A*dp_mask + (1-dp_mask)*mask_value
                # Dropout
                if self.training:
                    A = dropped_A()
                else:
                    A = self.identity(A)
                    
                A = nn.functional.softmax(A, dim=-1)
                
                mha_ops.append(torch.bmm(A, v))
            
            conc = torch.cat(mha_ops, dim=-1)
            proj = torch.matmul(conc, self.Wo[i,:,:])
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


class STraTS(nn.Module):
    def __init__(
            self, 
            D, 
            V, 
            d, 
            N, 
            he, 
            dropout, 
            with_text=False,
            text_encoder=None,
            text_encoder_name=None,
            text_linear_embed_dim=None,
            forecast=False, 
            return_embeddings=False,
            new_value_encoding=False,
            time_2_vec=False
        ):
        super(STraTS, self).__init__()
        total_parameters = 0
        cve_units = int(np.sqrt(d))

        self.with_text = with_text
        self.D = D
        self.return_embeddings = return_embeddings

        # Inputs: max_len * batch_size
        # To embed which feature is this value representing
        self.varis_stack = nn.Embedding(V+1, d)
        # num_params = sum(p.numel() for p in self.varis_stack.parameters())
        # print(f'varis_stack: {num_params}')
        # total_parameters += num_params

        if self.with_text:
            self.text_encoder = BertForRepresentation(text_encoder, text_encoder_name)
            # num_params = sum(p.numel() for p in self.text_encoder.parameters())
            # print(f'text_encoder: {num_params}')
            # total_parameters += num_params
            self.text_stack = TVE(text_linear_embed_dim, d)
            # num_params = sum(p.numel() for p in self.text_linear_stack.parameters())
            # print(f'text_linear_stack: {num_params}')
            # total_parameters += num_params

        # FFN to 'encode' the continuous values. Continuous Value Embedding (CVE)
        if new_value_encoding:
            self.values_stack = TVE(
                input_dim=2,
                hid_dim=cve_units, 
                output_dim=d
            )
        else:
            self.values_stack = CVE(
                hid_dim=cve_units, 
                output_dim=d
            )  
        # num_params = sum(p.numel() for p in self.values_stack.parameters())
        # print(f'values_stack: {num_params}')
        # total_parameters += num_params
        
        # FFN to 'encode' the continuous values. Continuous Value Embedding (CVE)
        if time_2_vec:
            self.times_stack = Time2Vec(
                output_dim=d
            )
        else:
            self.times_stack = CVE(
                hid_dim=cve_units, 
                output_dim=d
            )        
        # num_params = sum(p.numel() for p in self.times_stack.parameters())
        # print(f'times_stack: {num_params}')
        # total_parameters += num_params
        
        
        # Transformer Output = batch_size * max_len * d
        self.cont_stack = Transformer(
            d=d, 
            N=N, 
            h=he, 
            dk=None, 
            dv=None, 
            dff=None, 
            dropout=dropout, 
            epsilon=1e-07
        )
        # num_params = sum(p.numel() for p in self.cont_stack.parameters())
        # print(f'cont_stack: {num_params}')
        # total_parameters += num_params
        
        # Attention Output = batch_size * max_len * 1 
        self.attn_stack = Attention(
            d=d,
            hid_dim=2*d
        )
        # num_params = sum(p.numel() for p in self.attn_stack.parameters())
        # print(f'attn_stack: {num_params}')
        # total_parameters += num_params
        
        # Demographics Input : batch_size * D
        # Demographics Output: batch_size * d
        if self.D>0:
            self.demo_stack = nn.Sequential(
                nn.Linear(in_features=D, out_features=2*d),
                nn.Tanh(),
                nn.Linear(in_features=2*d, out_features=d),
                nn.Tanh()
            )
        # num_params = sum(p.numel() for p in self.demo_stack.parameters())
        # print(f'demo_stack: {num_params}')
        # total_parameters += num_params
        
        # Output Layer Inputs: Attention Weight * Time Series Embedding + Demographic Encoding = batch_size * (+d)
        if not self.return_embeddings:
            if forecast:
                self.output_stack = nn.Linear(in_features=d+d, out_features=V)
            else:
                if self.D>0:
                    self.output_stack = nn.Sequential(
                        nn.Linear(in_features=d+d, out_features=1),
                        nn.Sigmoid(),
                        nn.Flatten(start_dim=0)
                    )
                else:
                    self.output_stack = nn.Sequential(
                        nn.Linear(in_features=d, out_features=1),
                        nn.Sigmoid(),
                        nn.Flatten(start_dim=0)
                    )
        # num_params = sum(p.numel() for p in self.output_stack.parameters())
        # print(f'output_stack: {num_params}')
        # total_parameters += num_params
        
        # print(f'Total Parameters: {total_parameters}')
    
    def forward(self, demo, times, values, varis, text_values=None, text_attention_mask=None, text_times=None, text_varis=None):
        
        if self.D>0:
            demo_enc = self.demo_stack(demo)

        ts_varis_emb = self.varis_stack(varis)
        ts_values_emb = self.values_stack(values)
        ts_times_emb = self.times_stack(times)
        # print(f'ts_varis_emb: {ts_varis_emb.shape}')
        # print(f'ts_values_emb: {ts_values_emb.shape}')
        # print(f'ts_times_emb: {ts_times_emb.shape}')


        if self.with_text:
            text_varis_emb = self.varis_stack(text_varis)
            # print(f'text_varis_emb: {text_varis_emb.shape}')
            text_values_emb = self.text_encoder(text_values, text_attention_mask)
            # print(f'text_values_emb: {text_values_emb.shape}')
            text_values_emb = self.text_stack(text_values_emb)
            # print(f'text_values_emb: {text_values_emb.shape}')
            text_times_emb = self.times_stack(text_times)
            # print(f'text_times_emb: {text_times_emb.shape}')


            varis_emb = torch.cat([ts_varis_emb, text_varis_emb], dim=1)
            values_emb = torch.cat([ts_values_emb, text_values_emb], dim=1)
            times_emb = torch.cat([ts_times_emb, text_times_emb], dim=1)
        else:
            varis_emb, values_emb, times_emb = ts_varis_emb, ts_values_emb, ts_times_emb

        
        comb_emb = varis_emb + values_emb + times_emb
        # print(f'comb_emb: {comb_emb.shape}')
        if self.with_text:
            varis = torch.cat([varis, text_varis], dim=-1)
        mask = torch.clamp(varis, 0,1)
        # print(f'Mask: {mask.shape}')
        cont_emb = self.cont_stack(comb_emb, mask)
        # print(f'cont_emb: {cont_emb.shape}')
        
        # Calculating the weights for cont_emb
        attn_weights = self.attn_stack(cont_emb, mask)
        # print(f'attn_weights: {attn_weights.shape}')
        
        # Getting the weighted avg from the embeddings
        fused_emb = torch.sum(cont_emb * attn_weights, dim=-2)
        # print(f'fused_emb: {fused_emb.shape}')
        
        # Combining Time Series Embedding with Demographic Embeddings
        if self.D>0:
            conc = torch.cat([fused_emb, demo_enc], dim=-1)
        else:
            conc = fused_emb
        # print(f'conc: {conc.shape}')
        
        # Generating Output
        if self.return_embeddings:
            output = conc 
        else:
            output = self.output_stack(conc)
        # print(f'output: {output.shape}')
        
        return output
        