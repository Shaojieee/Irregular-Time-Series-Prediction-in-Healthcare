from module import CVE, Transformer, Attention, SpecialTransformer, MultiTimeAttention, WeightedValues, MultiModalAttention, ManualSpecialTransformer, ImputedTransformer, ImputedMultiTimeAttentionV1, ImputedMultiTimeAttentionV2

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda, Concatenate, Add, Reshape, Flatten
from tensorflow.keras import Model

# mtand_ffn_tf_fv_tfv
def build_modified_strats(D, max_len, V, d, N, he, dropout, forecast=False, with_demo=True):
    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d, activation='tanh')(demo)
        demo_enc = Dense(d, activation='tanh')(demo_enc)

    varis = Input(shape=(max_len,))
    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))
    varis_emb = Embedding(V+1, d)(varis)
    cve_units = int(np.sqrt(d))
    
    values_emb = CVE(cve_units, d)(values)
    times_emb = CVE(cve_units, d)(times)

    query = Add()([times_emb, varis_emb])
    key = Add()([varis_emb, values_emb])
    value = Add()([varis_emb, values_emb, times_emb]) # b, L, d

    mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L

    mtand_emb = SpecialTransformer(he, dk=None, dv=None, dff=None, dropout=dropout)([query, key, value], mask=mask)
    if N>0:
        cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(mtand_emb, mask=mask)
    else:
        cont_emb = mtand_emb

    attn_weights = Attention(2*d)(cont_emb, mask=mask)

    fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])
    if with_demo:
        conc = Concatenate(axis=-1)([fused_emb, demo_enc])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, times, values, varis], op)
        if forecast:
            fore_model = Model([demo, times, values, varis], fore_op)
            return [model, fore_model]
    else:
        fore_op = Dense(V)(fused_emb)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([times, values, varis], op)
        if forecast:
            fore_model = Model([times, values, varis], fore_op)
            return [model, fore_model]

    return model


# mtand_t_fv_tfv
# def build_modified_strats(D, max_len, V, d, N, he, dropout, forecast=False, with_demo=True):
#     if with_demo:
#         demo = Input(shape=(D,))
#         demo_enc = Dense(2*d, activation='tanh')(demo)
#         demo_enc = Dense(d, activation='tanh')(demo_enc)

#     varis = Input(shape=(max_len,))
#     values = Input(shape=(max_len,))
#     times = Input(shape=(max_len,))
#     varis_emb = Embedding(V+1, d)(varis)
#     cve_units = int(np.sqrt(d))
    
#     values_emb = CVE(cve_units, d)(values)
#     times_emb = CVE(cve_units, d)(times)

#     query = Add()([times_emb, varis_emb])
#     key = Add()([times_emb, varis_emb])
#     value = Add()([varis_emb, values_emb, times_emb]) # b, L, d

#     mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L

#     mtand_emb = MultiTimeAttention(he, dk=None, dv=None, dff=None, dropout=dropout)([query, key, value], mask=mask)
#     cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(mtand_emb, mask=mask)

#     attn_weights = Attention(2*d)(cont_emb, mask=mask)

#     fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])
#     if with_demo:
#         conc = Concatenate(axis=-1)([fused_emb, demo_enc])
#         fore_op = Dense(V)(conc)
#         op = Dense(1, activation='sigmoid')(fore_op)
#         model = Model([demo, times, values, varis], op)
#         if forecast:
#             fore_model = Model([demo, times, values, varis], fore_op)
#             return [model, fore_model]
#     else:
#         fore_op = Dense(V)(fused_emb)
#         op = Dense(1, activation='sigmoid')(fore_op)
#         model = Model([times, values, varis], op)
#         if forecast:
#             fore_model = Model([times, values, varis], fore_op)
#             return [model, fore_model]

#     return model


# with weighted at the start
# def build_modified_strats(D, max_len, V, d, N, he, dropout, forecast=False, with_demo=True):
#     if with_demo:
#         demo = Input(shape=(D,))
#         demo_enc = Dense(2*d, activation='tanh')(demo)
#         demo_enc = Dense(d, activation='tanh')(demo_enc)

#     varis = Input(shape=(max_len,))
#     values = Input(shape=(max_len,))
#     times = Input(shape=(max_len,))
#     varis_emb = Embedding(V+1, d)(varis)
#     cve_units = int(np.sqrt(d))
    
#     values_emb = CVE(cve_units, d)(values)
#     times_emb = CVE(cve_units, d)(times)

#     query = Add()([times_emb, varis_emb])
#     # query = times_emb
#     key = Add()([times_emb, varis_emb])
#     value = Add()([varis_emb, values_emb, times_emb]) # b, L, d

#     mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L

#     weighted_value = WeightedValues(hid_dim=d, dropout=dropout)([query, key, value], mask=mask)

#     cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(weighted_value, mask=mask)

#     attn_weights = Attention(2*d)(cont_emb, mask=mask)

#     fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])
#     if with_demo:
#         conc = Concatenate(axis=-1)([fused_emb, demo_enc])
#         fore_op = Dense(V)(conc)
#         op = Dense(1, activation='sigmoid')(fore_op)
#         model = Model([demo, times, values, varis], op)
#         if forecast:
#             fore_model = Model([demo, times, values, varis], fore_op)
#             return [model, fore_model]
#     else:
#         fore_op = Dense(V)(fused_emb)
#         op = Dense(1, activation='sigmoid')(fore_op)
#         model = Model([times, values, varis], op)
#         if forecast:
#             fore_model = Model([times, values, varis], fore_op)
#             return [model, fore_model]

#     return model


#with multimodal attention
# def build_modified_strats(D, max_len, V, d, N, he, dropout, forecast=False, with_demo=True):
#     if with_demo:
#         demo = Input(shape=(D,))
#         demo_enc = Dense(2*d, activation='tanh')(demo)
#         demo_enc = Dense(d, activation='tanh')(demo_enc)

#     varis = Input(shape=(max_len,))
#     values = Input(shape=(max_len,))
#     times = Input(shape=(max_len,))
#     varis_emb = Embedding(V+1, d)(varis)
#     cve_units = int(np.sqrt(d))
    
#     values_emb = CVE(cve_units, d)(values)
#     times_emb = CVE(cve_units, d)(times)

#     query = Add()([times_emb, varis_emb])
#     # query = times_emb
#     key = Add()([times_emb, varis_emb])
#     value = Add()([varis_emb, values_emb, times_emb]) # b, L, d

#     mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L

#     weighted_value = MultiModalAttention(hid_dim=d, dropout=dropout)([query, key, value], mask=mask)

#     cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(weighted_value, mask=mask)

#     attn_weights = Attention(2*d)(cont_emb, mask=mask)

#     fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])
#     if with_demo:
#         conc = Concatenate(axis=-1)([fused_emb, demo_enc])
#         fore_op = Dense(V)(conc)
#         op = Dense(1, activation='sigmoid')(fore_op)
#         model = Model([demo, times, values, varis], op)
#         if forecast:
#             fore_model = Model([demo, times, values, varis], fore_op)
#             return [model, fore_model]
#     else:
#         fore_op = Dense(V)(fused_emb)
#         op = Dense(1, activation='sigmoid')(fore_op)
#         model = Model([times, values, varis], op)
#         if forecast:
#             fore_model = Model([times, values, varis], fore_op)
#             return [model, fore_model]

#     return model


# with manual special transformer
def build_special_strats(D, max_len, V, d, N, he, dropout, forecast=False, with_demo=True):
    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d, activation='tanh')(demo)
        demo_enc = Dense(d, activation='tanh')(demo_enc)

    varis = Input(shape=(max_len,))
    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))
    varis_emb = Embedding(V+1, d)(varis)
    cve_units = int(np.sqrt(d))
    
    values_emb = CVE(cve_units, d)(values)
    times_emb = CVE(cve_units, d)(times)

    mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L

    mtand_emb = ManualSpecialTransformer(he, dk=None, dv=None, dff=None, dropout=dropout)([times_emb, values_emb, varis_emb], mask=mask)
    if N>0:
        cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(mtand_emb, mask=mask)
    else:
        cont_emb = mtand_emb

    attn_weights = Attention(2*d)(cont_emb, mask=mask)

    fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])
    if with_demo:
        conc = Concatenate(axis=-1)([fused_emb, demo_enc])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, times, values, varis], op)
        if forecast:
            fore_model = Model([demo, times, values, varis], fore_op)
            return [model, fore_model]
    else:
        fore_op = Dense(V)(fused_emb)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([times, values, varis], op)
        if forecast:
            fore_model = Model([times, values, varis], fore_op)
            return [model, fore_model]

    return model


def build_imputed_strats(D, max_len, V, d, N, he, dropout, forecast=False, with_demo=True):
    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d, activation='tanh')(demo)
        demo_enc = Dense(d, activation='tanh')(demo_enc)

    varis = Input(shape=(max_len,))
    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))
    imputed_mask = Input(shape=(max_len,))

    varis_emb = Embedding(V+1, d)(varis)
    cve_units = int(np.sqrt(d))
    
    values_emb = CVE(cve_units, d)(values)
    times_emb = CVE(cve_units, d)(times)
    comb_emb = Add()([varis_emb, values_emb, times_emb]) # b, L, d
    mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L
    cont_emb = ImputedTransformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(comb_emb, mask=mask, imputed_mask=imputed_mask)
    attn_weights = Attention(2*d)(cont_emb, mask=mask)
    fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])

    if with_demo:
        conc = Concatenate(axis=-1)([fused_emb, demo_enc])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, times, values, varis, imputed_mask], op)
        if forecast:
            fore_model = Model([demo, times, values, varis, imputed_mask], fore_op)
            return [model, fore_model]
    else:
        fore_op = Dense(V)(fused_emb)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([times, values, varis, imputed_mask], op)
        if forecast:
            fore_model = Model([times, values, varis, imputed_mask], fore_op)
            return [model, fore_model]

    return model


def build_strats(D, max_len, V, d, N, he, dropout, forecast=False, with_demo=True):
    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d, activation='tanh')(demo)
        demo_enc = Dense(d, activation='tanh')(demo_enc)

    varis = Input(shape=(max_len,))
    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))
    varis_emb = Embedding(V+1, d)(varis)
    cve_units = int(np.sqrt(d))
    
    values_emb = CVE(cve_units, d)(values)
    times_emb = CVE(cve_units, d)(times)
    comb_emb = Add()([varis_emb, values_emb, times_emb]) # b, L, d
    mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L
    cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(comb_emb, mask=mask)
    attn_weights = Attention(2*d)(cont_emb, mask=mask)
    fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])

    if with_demo:
        conc = Concatenate(axis=-1)([fused_emb, demo_enc])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, times, values, varis], op)
        if forecast:
            fore_model = Model([demo, times, values, varis], fore_op)
            return [model, fore_model]
    else:
        fore_op = Dense(V)(fused_emb)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([times, values, varis], op)
        if forecast:
            fore_model = Model([times, values, varis], fore_op)
            return [model, fore_model]

    return model


def build_mtand_strats(D, max_len, len_time_query, len_time_key, V, d_mtand, d_strats, N, he, dropout, forecast=False, with_demo=True):

    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d_strats, activation='tanh')(demo)
        demo_enc = Dense(d_strats, activation='tanh')(demo_enc)

    strats_varis = Input(shape=(max_len,))
    strats_values = Input(shape=(max_len,))
    strats_times = Input(shape=(max_len,))

    cve_units = int(np.sqrt(d_strats))
    time_encoder_block = CVE(cve_units,d_strats)
    values_encoder_block = CVE(cve_units, d_strats)
    varis_encoder_block = Embedding(V+1,d_strats)

    #STraTS
    strats_times_emb = time_encoder_block(strats_times)
    strats_values_emb = values_encoder_block(strats_values)
    strats_varis_emb = varis_encoder_block(strats_varis)

    strats_irregular_emb = Add()([strats_varis_emb, strats_values_emb, strats_times_emb]) # b, L, d
    strats_mask = Lambda(lambda x:K.clip(x,0,1))(strats_varis) # b, L
    strats_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(strats_irregular_emb, mask=strats_mask)
    strats_attn_weights = Attention(2*d_strats)(strats_emb, mask=strats_mask)
    strats_fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([strats_emb, strats_attn_weights])

    # mTAND model
    mtand_time_query = Input(shape=(len_time_query,))
    mtand_time_key = Input(shape=(len_time_key,))
    mtand_feature_matrix = Input(shape=(len_time_key, V))
    mtand_feature_mask = Input(shape=(len_time_key, V))

    cve_units = int(np.sqrt(d_mtand))
    mtand_time_encoder_block = CVE(cve_units,d_mtand)

    mtand_query_emb = mtand_time_encoder_block(mtand_time_query)
    mtand_key_emb = mtand_time_encoder_block(mtand_time_key)
    mtand_regular_emb = ImputedMultiTimeAttentionV1(h=8, dropout=dropout)([mtand_query_emb, mtand_key_emb, mtand_feature_matrix], mask=mtand_feature_mask) # b,time_query, d

    # mtand_fused_emb = mtand_regular_emb[:,-1,:]
    
    # mtand_emb = Transformer(1, he, dk=None, dv=None, dff=None, dropout=dropout)(mtand_regular_emb)
    # mtand_mask = Lambda(lambda x: K.zeros_like(x)[:,:,0])(mtand_regular_emb)
    
    mtand_attn_weights = Attention(2*d_mtand)(mtand_regular_emb)
    mtand_fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([mtand_regular_emb, mtand_attn_weights])
    # mtand_fused_emb = Dense(d_strats, activation='tanh')(mtand_regular_emb)
    
    
    # Combine
    # strats_unsqueeze = Reshape((1, d_strats))(strats_fused_emb) 
    # mtand_unsqueeze = Reshape((1, d_strats))(mtand_fused_emb) 
    # cont_emb = Concatenate(axis=-2)([strats_unsqueeze, mtand_unsqueeze]) # b, 2, d
    # print(cont_emb.shape)
    # attn_weights = Attention(2*d_strats)(cont_emb)
    # print(attn_weights.shape)
    # fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])
    # print(fused_emb.shape)


    # last_mtand_emb = mtand_cont_emb[:,-1,:]

    # print(last_mtand_emb.shape)
    # # Combining both mTAND and STraTS encoding
    # mtand_emb = Reshape((1, d))(last_mtand_emb) # b,1,d
    # print(mtand_emb.shape)
    # print(last_strats_emb.shape)
    # strats_emb = Reshape((1, d))(last_strats_emb) #b,1,d
    # print(strats_emb.shape)
    # fused_emb = Concatenate(axis=-2)([mtand_emb, strats_emb]) #b,2,d
    # print(fused_emb.shape)
    # fusion_attn_weights = Attention(2*d)(fused_emb)
    # final_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([fused_emb, fusion_attn_weights])
    # print(fusion_attn_weights.shape)


    if with_demo:
        conc = Concatenate(axis=-1)([mtand_fused_emb, strats_fused_emb, demo_enc])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], op)
        if forecast:
            fore_model = Model([demo, strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], fore_op)
            return [model, fore_model]
    else:
        conc = Concatenate(axis=-1)([mtand_fused_emb, strats_fused_emb])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], op)
        if forecast:
            fore_model = Model([strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], fore_op)
            return [model, fore_model]

    return model


def build_mtand(D, len_time_query, len_time_key, V, d_mtand, d_demo, N, he, dropout, forecast=False, with_demo=True):
    # STraTS model
    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d_demo, activation='tanh')(demo)
        demo_enc = Dense(d_demo, activation='tanh')(demo_enc)


    # mTAND model
    time_query = Input(shape=(len_time_query,))
    time_key = Input(shape=(len_time_key,))
    feature_matrix = Input(shape=(len_time_key, V))
    feature_mask = Input(shape=(len_time_key, V))

    cve_units = int(np.sqrt(d_mtand))
    time_encoder_block = CVE(cve_units, d_mtand)
    time_query_emb = time_encoder_block(time_query)
    time_key_emb = time_encoder_block(time_key)

    mtand_regular_emb = ImputedMultiTimeAttentionV1(h=he, dropout=dropout)([time_query_emb, time_key_emb, feature_matrix], mask=feature_mask) # b,time_query, d
    # cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(mtand_regular_emb)

    attn_weights = Attention(2*d_mtand)(mtand_regular_emb)
    fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([mtand_regular_emb, attn_weights])


    if with_demo:
        conc = Concatenate(axis=-1)([fused_emb, demo_enc])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, time_key, feature_matrix, feature_mask, time_query], op)
        if forecast:
            fore_model = Model([demo, time_key, feature_matrix, feature_mask, time_query], fore_op)
            return [model, fore_model]
    else:
        # conc = Concatenate(axis=-1)([cont_emb, last_strats_emb])
        fore_op = Dense(V)(fused_emb)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([time_key, feature_matrix, feature_mask, time_query], op)
        if forecast:
            fore_model = Model([time_key, feature_matrix, feature_mask, time_query], fore_op)
            return [model, fore_model]

    return model
