import torch.nn as nn
import torch 
import numpy as np
import math

from module import CVE, TVE, Time2Vec, MultiTimeAttention, TransformerEncoder, Attention, STraTS_Transformer, STraTS_MultiTimeAttention


class custom_STraTS(nn.Module):

    def __init__(
        self,
        D,
        V,
        d,
        N,
        he,
        dropout=0.1,
        time_2_vec=False,
        forecast=False
    ):
        super(custom_STraTS, self).__init__()
        self.D = D
        cve_units = int(np.sqrt(d))

        self.values_stack = CVE(
            hid_dim=cve_units,
            output_dim=d
        )

        if time_2_vec:
            self.times_stack = Time2Vec(
                output_dim=d
            )
        else:
            self.times_stack = CVE(
                hid_dim=cve_units,
                output_dim=d
            )
        
        self.varis_stack = nn.Embedding(V+1, d)

        # self.mTAND = MultiTimeAttention(
        #     input_dim=d, 
        #     hid_dim=d,
        #     output_dim=d,
        #     num_heads=he,
        #     dropout=dropout,
        # )

        # self.CTE = TransformerEncoder(
        #     embed_dim=d, 
        #     num_heads=he, 
        #     layers=N,
        #     attn_dropout=dropout, 
        #     relu_dropout=dropout, 
        #     res_dropout=dropout,     
        #     embed_dropout=dropout, 
        #     attn_mask=False,
        #     q_seq_len=None, 
        #     kv_seq_len=None
        # )

        self.mTAND = STraTS_MultiTimeAttention(
            d=d,
            h=he,
            dk=None, 
            dv=None, 
            dff=None, 
            dropout=dropout, 
            epsilon=1e-07
        )

        self.CTE = STraTS_Transformer(
            d=d, 
            N=N, 
            h=he, 
            dk=None, 
            dv=None, 
            dff=None, 
            dropout=dropout, 
            epsilon=1e-07
        )

        self.atten_stack = Attention(
            d=d,
            hid_dim=2*d
        )

        if self.D>0:
            self.demo_stack = nn.Sequential(
                nn.Linear(in_features=D, out_features=2*d),
                nn.Tanh(),
                nn.Linear(in_features=2*d, out_features=d),
                nn.Tanh()
            )
        if forecast and self.D>0:
            self.output_stack = nn.Linear(in_features=d+d, out_features=V)
        elif forecast and self.D==0:
            self.output_stack = nn.Linear(in_features=d, out_features=V)
        elif self.D>0:
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
    
    def forward(self, demo, times, values, varis):

        
        ts_varis_emb = self.varis_stack(varis)
        ts_values_emb = self.values_stack(values)
        ts_times_emb = self.times_stack(times)
        # print(f'ts_varis_emb: {ts_varis_emb.shape}')
        # print(f'ts_values_emb: {ts_values_emb.shape}')
        # print(f'ts_times_emb: {ts_times_emb.shape}')

        mask = torch.clamp(varis, 0, 1)
        # print(f'mask: {mask.shape}')

        query_key_emb = ts_varis_emb + ts_times_emb
        value_emb = ts_values_emb + query_key_emb

        time_atten_values_emb = self.mTAND(
            query=query_key_emb,
            key=query_key_emb,
            value=value_emb,
            mask=mask
        )


        # print(f'time_atten: {time_atten_values_emb.shape}')
        # time_atten = time_atten.transpose(0,1)
        # print(f'time_atten: {time_atten.shape}')

        # comb_emb = time_atten_values_emb + ts_varis_emb + ts_times_emb
        comb_emb = time_atten_values_emb

        CTE_emb = self.CTE(comb_emb, mask)
        # print(f'CTE_emb: {CTE_emb.shape}')
        # CTE_emb = CTE_emb.transpose(0,1)
        # print(f'CTE_emb: {CTE_emb.shape}')

        # Calculating the weights for cont_emb
        attn_weights = self.atten_stack(CTE_emb, mask)
        # print(f'attn_weights: {attn_weights.shape}')
        
        # Getting the weighted avg from the embeddings
        fused_emb = torch.sum(CTE_emb * attn_weights, dim=-2)
        # print(f'fused_emb: {fused_emb.shape}')

        if self.D>0:
            demo_enc = self.demo_stack(demo)
            conc = torch.cat([fused_emb, demo_enc], dim=-1)
        else:
            conc = fused_emb
        
        # print(f'conc: {conc.shape}')


        output = self.output_stack(conc)

        return output