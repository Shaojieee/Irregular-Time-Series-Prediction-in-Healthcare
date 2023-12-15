from model import STraTS
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel

def load_Bert(text_encoder_model):
    if text_encoder_model!=None:
        if text_encoder_model== 'BioBert':
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            BioBert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # elif text_encoder_model=="bioRoberta":
        #     config = AutoConfig.from_pretrained("allenai/biomed_roberta_base", num_labels=args.num_labels)
        #     tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
        #     BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")

        elif text_encoder_model=="Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            BioBert = BertModel.from_pretrained("bert-base-uncased")

        elif text_encoder_model=="bioLongformer":
            tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
            BioBert = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")

        else:
            raise ValueError("text_encoder_model should be BioBert,bioRoberta,bioLongformer or Bert")

    BioBertConfig = BioBert.config
    return BioBert, BioBertConfig, tokenizer


class BertForRepresentation(nn.Module):

    def __init__(self, model, model_name):
        super().__init__()

        self.model = model
        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
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

    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)

            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn=F.dropout(p_attn, p=dropout, training=self.training)
            # p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        # import pdb; pdb.set_trace()
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)

        return self.linears[-1](x)


class Time2Vec(nn.Module):
    def __init__(output_dim):
        super(Time2Vec, self).__init__()

        self.periodic = nn.Linear(1, output_dim-1)
        self.linear = nn.Linear(1, 1)

    def forward(self, time):
        time = time.unsqueeze(-1)
        periodic_out = torch.sin(self.periodic(time))
        linear_out = self.linear(time)
        return torch.cat([linear_out, periodic_out],-1)


class TransformerCrossEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerCrossEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False,q_seq_len_1=None,q_seq_len_2=None):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)

        self.q_seq_len_1=q_seq_len_1
        self.q_seq_len_2=q_seq_len_2
        # self.intermediate=intermediate
        self.embed_positions_q_1=nn.Embedding(self.q_seq_len_1,embed_dim,padding_idx=0)
        nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)

        if self.q_seq_len_2!= None:
            self.embed_positions_q_2=nn.Embedding(self.q_seq_len_2,embed_dim,padding_idx=0)
            nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)

            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1,self.embed_positions_q_2])
        else:
            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1,self.embed_positions_q_1,])


        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerCrossEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x_in_list):
        """
        Args:
            x_in_list (list of FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the list of last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

        """

        # import pdb;
        # pdb.set_trace()
        x_list=x_in_list
        length_x1 = x_list[0].size(0) # (length,Batch_size,input_dim)
        length_x2 = x_list[1].size(0)
        x_list = [ self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            position_x1 = torch.tensor(torch.arange(length_x1),dtype=torch.long)
            position_x2 = torch.tensor(torch.arange(length_x2),dtype=torch.long)
            positions=[position_x1 ,position_x2]
            x_list=[ l(position_x).unsqueeze(0).transpose(0,1) +x for l, x,position_x in zip(self.embed_positions_q, x_list,positions)]
              # Add positional embedding
        x_list[0]=F.dropout(x_list[0], p=self.dropout, training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.dropout, training=self.training)

        # encoder layers

        # x_low_level=None


        for layer in self.layers:
            x_list= layer(x_list) #proj_x_txt, proj_x_ts


        if self.normalize:
            x_list=[ l(x)  for l, x in zip(self.layer_norm, x_list)]
        return x_list


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                     attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.self_attns = nn.ModuleList([MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        ) for _ in range(2)])

        self.post_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])


        self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.cross_attn_1 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.cross_attn_2 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.post_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        self.fc1 =  nn.ModuleList([nn.Linear(self.embed_dim, 4*self.embed_dim) for _ in range(2)])  # The "Add & Norm" part in the paper
        self.fc2 = nn.ModuleList([nn.Linear(4*self.embed_dim, self.embed_dim) for _ in range(2)])
        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])


    def forward(self, x_list):
        """
        Args:
            x (List of Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            list of encoded output of shape `(batch, src_len, embed_dim)`
        """
        ###self attn
        residual = x_list

        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]

        output= [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]

        x_list=[ x for x, _ in output]

        x_list[0]=F.dropout(x_list[0], p=self.res_dropout , training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.res_dropout , training=self.training)

        x_list = [r + x  for r, x in zip(residual, x_list) ]
        # x_list = [l(x) for l, x in zip(self.post_self_attn_layer_norm, x_list)]

        #### cross attn

        residual=x_list
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        x_txt,x_ts=  x_list #proj_x_txt, proj_x_ts

        # cross: ts -> txt
        x_ts_to_txt,_=self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
        # cross:  txt->ts
        x_txt_to_ts,_=self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)

        # else:
        #     x_low_level = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_low_level)]
        #     x_txt_low,x_ts_low=  x_low_level
        #     # cross: ts -> txt
        #     x_ts_to_txt,_=self.cross_attn_1(query=x_txt, key=x_ts_low, value=x_ts_low)
        #     # cross:  txt->ts
        #     x_txt_to_ts,_=self.cross_attn_2(query=x_ts, key=x_txt_low, value=x_txt_low)


        x_ts_to_txt  = F.dropout(x_ts_to_txt, p=self.res_dropout, training=self.training)
        x_txt_to_ts  = F.dropout(x_txt_to_ts, p=self.res_dropout, training=self.training)

        x_list = [r+ x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

        # x_list = [l(x) for l, x in zip(self.post_encoder_attn_layer_norm, x_list)]

        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]

        x_list[0]=F.dropout(x_list[0], p=self.relu_dropout , training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.relu_dropout , training=self.training)

        x_list = [l(x) for l, x in zip(self.fc2, x_list)]

        x_list[0]=F.dropout(x_list[0], p=self.res_dropout, training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.res_dropout, training=self.training)

        x_list = [r + x  for r, x in zip(residual, x_list) ]

        # x_list = [l(x) for l, x in zip(self.post_ffn_layer_norm, x_list)]


        return x_list


class STraTS_text(nn.Module):
    def __init__(
        self, 
        D, # No. of static variables
        V, # No. of variables / features
        d, # Input size of attention layer
        N, # No. of Encoder blocks
        he, # No. of heads in multi headed encoder blocks
        dropout,
        text_seq_num, # No. of notes
        text_atten_embed_dim,
        text_time_embedding_dim,
        period_length,
        text_encoder_model,
        text_encoder_model_name,
        num_cross_layers, # No. of cross layers with multi modal
        num_cross_heads, # No. of heads in cross transformaer
        cross_dropout,
        output_dim
    ):
        super(STraTS_text, self).__init__()

        # Numerical time series block
        self.STraTS_block = STraTS(D, V, d, N, he, dropout, return_embeddings=True)

        # Text data block
        self.text_atten_embed_dim = text_atten_embed_dim
        self.text_seq_num = text_seq_num
        self.time_2_vec = Time2Vec(time_embedding_dim)
        self.time_query = torch.linspace(0, 1.0, period_length)
        self.text_encoder_block = BertForRepresentation(text_encoder_model, text_encoder_model_name)
        # mTAND_txt module
        self.mTAND_txt = MultiTimeAttention(768, self.text_atten_embed_dim, text_time_embedding_dim, num_heads=8)

        # Self and cross attention 
        self.trans_self_cross_ts_txt = TransformerCrossEncoder(
            embed_dim=text_atten_embed_dim,
            num_heads=num_cross_heads,
            layers=num_cross_layers,
            attn_dropout=cross_dropout,
            relu_dropout=cross_dropout,
            res_dropout=cross_dropout,
            embed_dropout=cross_dropout,
            attn_mask=False,
            q_seq_len_1=period_length
        )


        self.ts_embed_dim = 2*d if D>0 else d
        # Output linear stack
        self.linear_stack = nn.Sequential(
            nn.Linear(self.ts_embed_dim+self.text_atten_embed_dim, self.ts_embed_dim+self.text_atten_embed_dim),
            nn.ReLU(),
            nn.Dropout(p=cross_dropout),
            nn.Linear(self.ts_embed_dim+self.text_atten_embed_dim, self.ts_embed_dim+self.text_atten_embed_dim),
        )

        self.output_layer = nn.Linear(self.ts_embed_dim+self.text_atten_embed_dim, output_dim)


    def forward(self, demo, times, values, varis, text_tokens, text_attention_mask, text_times, text_time_mask):
        numeric_encoding = self.STraTS_block(demo, times, values, varis)

        text_encoding_value = self.text_encoder_block(text_tokens, text_attention_mask)

        time_key = self.time_2_vec(text_times)
        time_query = self.time_2_vec(self.time_query.unsqueeze(0))

        text_encoding=self.mTAND_txt(time_query, time_key, text_encoding_value, text_time_mask)
        text_encoding=text_encoding.transpose(0, 1)

        hiddens = self.trans_self_cross_ts_txt([text_encoding, numeric_encoding])
        h_txt_with_ts, h_ts_with_txt = hiddens
        last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)

        last_hs_proj = self.linear_stack(last_hs)
        last_hs_proj += last_hs
        output = self.output_layer(last_hs_proj)

        return output