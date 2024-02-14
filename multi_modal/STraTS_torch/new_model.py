import torch.nn as nn
import torch 
import numpy as np
import math

from module import CVE, TVE, Time2Vec, MultiTimeAttention, Attention, STraTS_Transformer, STraTS_MultiTimeAttention, STraTS_SpecialTransformer


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
        total_parameters = 0

        self.values_stack = CVE(
            hid_dim=cve_units,
            output_dim=d
        )
        num_params = sum(p.numel() for p in self.values_stack.parameters())
        print(f'values_stack: {num_params}')
        total_parameters += num_params

        if time_2_vec:
            self.times_stack = Time2Vec(
                output_dim=d
            )
        else:
            self.times_stack = CVE(
                hid_dim=cve_units,
                output_dim=d
            )
        num_params = sum(p.numel() for p in self.times_stack.parameters())
        print(f'times_stack: {num_params}')
        total_parameters += num_params
        
        self.varis_stack = nn.Embedding(V+1, d)
        num_params = sum(p.numel() for p in self.varis_stack.parameters())
        print(f'varis_stack: {num_params}')
        total_parameters += num_params

        # self.mTAND = MultiTimeAttention(
        #     input_dim=d, 
        #     output_dim=d,
        #     hid_dim=16,
        #     num_heads=he, 
        #     dropout=0.2
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

        # self.mTAND = STraTS_SpecialTransformer(
        #     d=d,
        #     h=he,
        #     dk=None, 
        #     dv=None, 
        #     dff=None, 
        #     dropout=dropout, 
        #     epsilon=1e-07
        # )

        num_params = sum(p.numel() for p in self.mTAND.parameters())
        print(f'mTAND: {num_params}')
        total_parameters += num_params

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
        num_params = sum(p.numel() for p in self.CTE.parameters())
        print(f'CTE: {num_params}')
        total_parameters += num_params

        self.atten_stack = Attention(
            d=d,
            hid_dim=2*d
        )
        num_params = sum(p.numel() for p in self.atten_stack.parameters())
        print(f'atten_stack: {num_params}')
        total_parameters += num_params

        if self.D>0:
            self.demo_stack = nn.Sequential(
                nn.Linear(in_features=D, out_features=2*d),
                nn.Tanh(),
                nn.Linear(in_features=2*d, out_features=d),
                nn.Tanh()
            )
            num_params = sum(p.numel() for p in self.demo_stack.parameters())
            print(f'demo_stack: {num_params}')
            total_parameters += num_params

        if forecast and self.D>0:
            self.output_stack = nn.Linear(in_features=d+d, out_features=V)
        elif forecast and self.D==0:
            self.output_stack = nn.Linear(in_features=d, out_features=V)
        elif self.D>0:
            self.output_stack = nn.Sequential(
                    nn.Linear(in_features=d+d, out_features=V),
                    nn.Linear(in_features=V, out_features=1),
                    nn.Sigmoid(),
                    nn.Flatten(start_dim=0)
            )
        else:
            self.output_stack = nn.Sequential(
                    nn.Linear(in_features=d, out_features=V),
                    nn.Linear(in_features=V, out_features=1),
                    nn.Sigmoid(),
                    nn.Flatten(start_dim=0)
            )
        
        num_params = sum(p.numel() for p in self.output_stack.parameters())
        print(f'output_stack: {num_params}')
        total_parameters += num_params

        print(f'Total Parameter: {total_parameters}')
    
    def forward(self, demo, times, values, varis):

        
        ts_varis_emb = self.varis_stack(varis)
        ts_values_emb = self.values_stack(values)
        ts_times_emb = self.times_stack(times)
        # print(f'ts_varis_emb: {ts_varis_emb.shape}')
        # print(f'ts_values_emb: {ts_values_emb.shape}')
        # print(f'ts_times_emb: {ts_times_emb.shape}')

        mask = torch.clamp(varis, 0, 1)
        # print(f'mask: {mask.shape}')

        query_emb = ts_times_emb
        key_emb = ts_values_emb + ts_times_emb
        value_emb = ts_values_emb + ts_varis_emb + ts_times_emb

        time_atten_values_emb = self.mTAND(
            query=query_emb,
            key=key_emb,
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