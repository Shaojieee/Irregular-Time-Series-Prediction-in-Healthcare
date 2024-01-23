import torch.nn as nn
import torch
import numpy as np
from module import CVE, TVE, Time2Vec, Attention, STraTS_Transformer



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
        self.cont_stack = STraTS_Transformer(
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
        # num_params = sum(p.numel() for p in self.output_stack.parameters())
        # print(f'output_stack: {num_params}')
        # total_parameters += num_params
        
        # print(f'Total Parameters: {total_parameters}')
    
    def forward(self, demo, times, values, varis):
        
        if self.D>0:
            demo_enc = self.demo_stack(demo)

        
        ts_varis_emb = self.varis_stack(varis)
        ts_values_emb = self.values_stack(values)
        ts_times_emb = self.times_stack(times)
        # print(f'ts_varis_emb: {ts_varis_emb.shape}')
        # print(f'ts_values_emb: {ts_values_emb.shape}')
        # print(f'ts_times_emb: {ts_times_emb.shape}')


        varis_emb, values_emb, times_emb = ts_varis_emb, ts_values_emb, ts_times_emb

    
        comb_emb = varis_emb + values_emb + times_emb

        # print(f'comb_emb: {comb_emb.shape}')
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


