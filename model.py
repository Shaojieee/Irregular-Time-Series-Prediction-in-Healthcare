from module import CVE, Transformer, Attention, ImputedMultiTimeAttention

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Input, Dense, Lambda, Concatenate, Add
from tensorflow.keras import Model


# Default STraTS implementation 
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

# STraTS with mTAND
def build_mtand_strats(D, V, max_len, d_strats, N_strats, he_strats, dropout_strats, len_time_query, len_time_key, d_mtand, N_mtand, he_mtand, dropout_mtand, forecast=False, with_demo=True):

    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d_strats, activation='tanh')(demo)
        demo_enc = Dense(d_strats, activation='tanh')(demo_enc)

    # STraTS inputs
    strats_varis = Input(shape=(max_len,))
    strats_values = Input(shape=(max_len,))
    strats_times = Input(shape=(max_len,))

    # STraTS encoders
    cve_units = int(np.sqrt(d_strats))
    time_encoder_block = CVE(cve_units,d_strats)
    values_encoder_block = CVE(cve_units, d_strats)
    varis_encoder_block = Embedding(V+1,d_strats)

    # Encoding STraTS inputs
    strats_times_emb = time_encoder_block(strats_times)
    strats_values_emb = values_encoder_block(strats_values)
    strats_varis_emb = varis_encoder_block(strats_varis)
    strats_irregular_emb = Add()([strats_varis_emb, strats_values_emb, strats_times_emb]) # b, L, d # Triplet embeddings

    # Generating observation embeddings
    strats_mask = Lambda(lambda x:K.clip(x,0,1))(strats_varis) # b, L # Mask
    strats_emb = Transformer(N_strats, he_strats, dk=None, dv=None, dff=None, dropout=dropout_strats)(strats_irregular_emb, mask=strats_mask)
    strats_attn_weights = Attention(2*d_strats)(strats_emb, mask=strats_mask)
    strats_fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([strats_emb, strats_attn_weights])

    # mTAND inputs
    mtand_time_query = Input(shape=(len_time_query,))
    mtand_time_key = Input(shape=(len_time_key,))
    mtand_feature_matrix = Input(shape=(len_time_key, V))
    mtand_feature_mask = Input(shape=(len_time_key, V))

    # mTAND encoders
    cve_units = int(np.sqrt(d_mtand))
    mtand_time_encoder_block = CVE(cve_units,d_mtand)

    # Encoding mTAND inputs
    mtand_query_emb = mtand_time_encoder_block(mtand_time_query)
    mtand_key_emb = mtand_time_encoder_block(mtand_time_key)

    # Imputing mTAND
    mtand_value = Concatenate(axis=-1)([mtand_feature_matrix, mtand_feature_mask])
    mtand_value_mask = Concatenate(axis=-1)([mtand_feature_mask, mtand_feature_mask])
    mtand_regular_emb = ImputedMultiTimeAttention(h=he_mtand, dropout=dropout_mtand)([mtand_query_emb, mtand_key_emb, mtand_value], mask=mtand_value_mask) # b,time_query, d

    
    if N_mtand>0:
        mtand_emb = Transformer(N_mtand, he_mtand, dk=None, dv=None, dff=None, dropout=dropout_strats)(mtand_regular_emb)
    else:
        mtand_emb = mtand_regular_emb
    
    mtand_attn_weights = Attention(2*d_mtand)(mtand_emb)
    mtand_fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([mtand_emb, mtand_attn_weights])

    if with_demo:
        conc = Concatenate(axis=-1)([mtand_fused_emb, strats_fused_emb, demo_enc])
        # Output FFNs
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], op)
        if forecast:
            fore_model = Model([demo, strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], fore_op)
            return [model, fore_model]
    else:
        conc = Concatenate(axis=-1)([mtand_fused_emb, strats_fused_emb])
        # Output FFNs
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], op)
        if forecast:
            fore_model = Model([strats_times, strats_values, strats_varis, mtand_time_key, mtand_feature_matrix, mtand_feature_mask, mtand_time_query], fore_op)
            return [model, fore_model]

    return model

# mTAND with FFN
def build_mtand(D, len_time_query, len_time_key, V, d_mtand, d_demo, N, he, dropout, forecast=False, with_demo=True):
    # STraTS model
    if with_demo:
        demo = Input(shape=(D,))
        demo_enc = Dense(2*d_demo, activation='tanh')(demo)
        demo_enc = Dense(d_demo, activation='tanh')(demo_enc)


    # mTAND inputs
    time_query = Input(shape=(len_time_query,))
    time_key = Input(shape=(len_time_key,))
    feature_matrix = Input(shape=(len_time_key, V))
    feature_mask = Input(shape=(len_time_key, V))

    # mTAND encoders
    cve_units = int(np.sqrt(d_mtand))
    time_encoder_block = CVE(cve_units, d_mtand)

    # Encoding mTAND inputs
    time_query_emb = time_encoder_block(time_query)
    time_key_emb = time_encoder_block(time_key)

    # Imputing inputs
    mtand_regular_emb = ImputedMultiTimeAttention(h=he, dropout=dropout)([time_query_emb, time_key_emb, feature_matrix], mask=feature_mask) # b,time_query, d

    if N>0: # Transfomer to encode mtand output
        mtand_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(mtand_regular_emb)
    else:
        mtand_emb = mtand_regular_emb

    # Fuse mtand emb
    attn_weights = Attention(2*d_mtand)(mtand_emb)
    fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([mtand_emb, attn_weights])


    if with_demo:
        conc = Concatenate(axis=-1)([fused_emb, demo_enc])
        fore_op = Dense(V)(conc)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([demo, time_key, feature_matrix, feature_mask, time_query], op)
        if forecast:
            fore_model = Model([demo, time_key, feature_matrix, feature_mask, time_query], fore_op)
            return [model, fore_model]
    else:
        fore_op = Dense(V)(fused_emb)
        op = Dense(1, activation='sigmoid')(fore_op)
        model = Model([time_key, feature_matrix, feature_mask, time_query], op)
        if forecast:
            fore_model = Model([time_key, feature_matrix, feature_mask, time_query], fore_op)
            return [model, fore_model]

    return model
