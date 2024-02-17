import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Add, Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow import nn
import math 


    
class CVE(Layer):
    def __init__(self, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(CVE, self).__init__()
        
    def build(self, input_shape): 
        self.W1 = self.add_weight(name='CVE_W1',
                            shape=(1, self.hid_units),
                            initializer='glorot_uniform',
                            trainable=True)
        self.b1 = self.add_weight(name='CVE_b1',
                            shape=(self.hid_units,),
                            initializer='zeros',
                            trainable=True)
        self.W2 = self.add_weight(name='CVE_W2',
                            shape=(self.hid_units, self.output_dim),
                            initializer='glorot_uniform',
                            trainable=True)
        super(CVE, self).build(input_shape)
        
    def call(self, x):
        if len(x.shape)==2:
            x = K.expand_dims(x, axis=-1)
        x = K.dot(K.tanh(K.bias_add(K.dot(x, self.W1), self.b1)), self.W2)
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)
    
    
class Attention(Layer):
    
    def __init__(self, hid_dim):
        self.hid_dim = hid_dim
        super(Attention, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        self.W = self.add_weight(shape=(d, self.hid_dim), name='Att_W',
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.hid_dim,), name='Att_b',
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(self.hid_dim,1), name='Att_u',
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)
        
    def call(self, x, mask=None, mask_value=-1e30):
        attn_weights = K.dot(K.tanh(K.bias_add(K.dot(x,self.W), self.b)), self.u)
        if mask is not None:
            mask = K.expand_dims(mask, axis=-1)
            attn_weights = mask*attn_weights + (1-mask)*mask_value
        attn_weights = K.softmax(attn_weights, axis=-2)
        return attn_weights
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


# Self Attention with Layer Norma and FFN 
class Transformer(Layer):
    
    def __init__(self, N=2, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(Transformer, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.N, self.h, d, self.dv), name='Wv',
                                 initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.N, self.dv*self.h, d), name='Wo',
                                 initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(self.N, d, self.dff), name='W1',
                                 initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.N, self.dff), name='b1',
                                 initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.N, self.dff, d), name='W2',
                                 initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(self.N, d), name='b2',
                                 initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2*self.N,), name='gamma',
                                 initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2*self.N,), name='beta',
                                 initializer='zeros', trainable=True)
        super(Transformer, self).build(input_shape)
        
    def call(self, x, mask=None, mask_value=-1e30):
        if mask is not None:
            mask = K.expand_dims(mask, axis=-2)
        for i in range(self.N):
            # MHA
            mha_ops = []
            for j in range(self.h):
                q = K.dot(x, self.Wq[i,j,:,:])
                k = K.permute_dimensions(K.dot(x, self.Wk[i,j,:,:]), (0,2,1))
                v = K.dot(x, self.Wv[i,j,:,:])
                A = K.batch_dot(q,k) # b, L, L
                # Mask unobserved steps.
                if mask is not None:
                    A = mask*A + (1-mask)*mask_value
                # Mask for attention dropout.
                def dropped_A():
                    dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                    return A*dp_mask + (1-dp_mask)*mask_value
                A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
                A = K.softmax(A, axis=-1)
                mha_ops.append(K.batch_dot(A,v)) # b, L, dv
            conc = K.concatenate(mha_ops, axis=-1)
            proj = K.dot(conc , self.Wo[i,:,:])
            # Dropout.
            proj = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),\
                                       lambda: array_ops.identity(proj))
            # Add & LN
            x = x+proj
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i] + self.beta[2*i]
            # FFN
            ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1[i,:,:]), self.b1[i,:])), 
                           self.W2[i,:,:]), self.b2[i,:,])
            # Dropout.
            ffn_op = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),\
                                       lambda: array_ops.identity(ffn_op))
            # Add & LN
            x = x+ffn_op
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i+1] + self.beta[2*i+1]            
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape


# Attention with Layer Norm and FFN
class SpecialTransformer(Layer):
    
    def __init__(self, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.h, self.dk, self.dv, self.dff, self.dropout = h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(SpecialTransformer, self).__init__()

    def build(self, input_shape):
        d = input_shape[0].as_list()[-1]
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.h, d, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.h, d, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.h, d, self.dv), name='Wv',
                                 initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.dv*self.h, d), name='Wo',
                                 initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(d, self.dff), name='W1',
                                 initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.dff), name='b1',
                                 initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.dff, d), name='W2',
                                 initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(d), name='b2',
                                 initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2,), name='gamma',
                                 initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2,), name='beta',
                                 initializer='zeros', trainable=True)
        super(SpecialTransformer, self).build(input_shape)
        
    def call(self, inputs, mask, mask_value=-1e30):
        query, key, value = inputs
        mask = K.expand_dims(mask, axis=-2)
        # MHA
        mha_ops = []
        for j in range(self.h):
            q = K.dot(query, self.Wq[j,:,:])
            k = K.permute_dimensions(K.dot(key, self.Wk[j,:,:]), (0,2,1))
            v = K.dot(value, self.Wv[j,:,:])
            A = K.batch_dot(q,k)

            # Mask unobserved steps.
            A = mask*A + (1-mask)*mask_value
            # Mask for attention dropout.
            def dropped_A():
                dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                return A*dp_mask + (1-dp_mask)*mask_value
            A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
            A = K.softmax(A, axis=-1)
            mha_ops.append(K.batch_dot(A,v))
        conc = K.concatenate(mha_ops, axis=-1)   
        proj = K.dot(conc , self.Wo)
        # Dropout.
        proj = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),\
                                   lambda: array_ops.identity(proj))
        # Add & LN
        x = value+proj
        mean = K.mean(x, axis=-1, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        x = (x - mean) / std
        x = x*self.gamma[0] + self.beta[0]
        # FFN
        ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1), self.b1)), 
                       self.W2), self.b2)
        # Dropout.
        ffn_op = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),\
                                   lambda: array_ops.identity(ffn_op))
        # Add & LN
        x = x+ffn_op
        mean = K.mean(x, axis=-1, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        x = (x - mean) / std
        x = x*self.gamma[1] + self.beta[1]   
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape


# Attention with Layer Norm
class MultiTimeAttention(Layer):
    
    def __init__(self, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.h, self.dk, self.dv, self.dff, self.dropout = h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(MultiTimeAttention, self).__init__()

    def build(self, input_shape):
        d = input_shape[0].as_list()[-1]
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.h, d, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.h, d, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.h, d, self.dv), name='Wv',
                                 initializer='glorot_uniform', trainable=True)
        # self.Wo = self.add_weight(shape=(self.dv*self.h, d), name='Wo',
        #                          initializer='glorot_uniform', trainable=True)
        self.gamma = self.add_weight(shape=(1,), name='gamma',
                                 initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(1,), name='beta',
                                 initializer='zeros', trainable=True)
        super(MultiTimeAttention, self).build(input_shape)
        
    def call(self, inputs, mask, mask_value=-1e30):
        query, key, value = inputs
        mask = K.expand_dims(mask, axis=-2)
        # MHA
        mha_ops = []
        for j in range(self.h):
            q = K.dot(query, self.Wq[j,:,:])
            k = K.permute_dimensions(K.dot(key, self.Wk[j,:,:]), (0,2,1))
            v = K.dot(value, self.Wv[j,:,:])
            A = K.batch_dot(q,k)

            # Mask unobserved steps.
            A = mask*A + (1-mask)*mask_value
            # Mask for attention dropout.
            def dropped_A():
                dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                return A*dp_mask + (1-dp_mask)*mask_value
            A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
            A = K.softmax(A, axis=-1)
            mha_ops.append(K.batch_dot(A,v))
        conc = K.concatenate(mha_ops, axis=-1)   
        
        # Dropout.
        conc = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(conc, rate=self.dropout)),\
                                   lambda: array_ops.identity(conc))
        # LN
        mean = K.mean(conc, axis=-1, keepdims=True)
        variance = K.mean(K.square(conc - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        x = (conc - mean) / std
        x = x*self.gamma[0] + self.beta[0] 

        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape


# Attention but no Wv on value
class WeightedValues(Layer):
    def __init__(self, hid_dim, dropout=0):
        self.hid_dim, self.dropout = hid_dim, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(WeightedValues, self).__init__()

    def build(self, input_shape):
        d = input_shape[0].as_list()[-1]

        self.Wq = self.add_weight(shape=(d, self.hid_dim), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(d, self.hid_dim), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        super(WeightedValues, self).build(input_shape)
        
    def call(self, inputs, mask, mask_value=-1e30):
        query, key, value = inputs
        mask = K.expand_dims(mask, axis=-2)
        q = K.dot(query, self.Wq)
        k = K.permute_dimensions(K.dot(key, self.Wk), (0,2,1))
        A = K.batch_dot(q,k)

        # Mask unobserved steps.
        A = mask*A + (1-mask)*mask_value
        # Mask for attention dropout.
        def dropped_A():
            dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
            return A*dp_mask + (1-dp_mask)*mask_value
        A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))

        A = K.softmax(A, axis=-1)
        value = K.batch_dot(A, value)
        
        return value
        
    def compute_output_shape(self, input_shape):
        return input_shape


# parallel way of computing attention
class MultiModalAttention(Layer):
    def __init__(self, hid_dim, dropout=0):
        self.hid_dim, self.dropout = hid_dim, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(MultiModalAttention, self).__init__()

    def build(self, input_shape):
        d = input_shape[0].as_list()[-1]

        self.Wq = self.add_weight(shape=(d, self.hid_dim), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(d, self.hid_dim), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        # self.Wv = self.add_weight(shape=(d, self.hid_dim), name='Wv',
        #                          initializer='glorot_uniform', trainable=True)
        # self.gamma = self.add_weight(shape=(1,), name='gamma',
        #                          initializer='ones', trainable=True)
        # self.beta = self.add_weight(shape=(1,), name='beta',
        #                          initializer='zeros', trainable=True)
        super(MultiModalAttention, self).build(input_shape)
        
    def call(self, inputs, mask, mask_value=-1e30):
        query, key, value = inputs
        mask = K.expand_dims(mask, axis=-2)
        q = K.dot(query, self.Wq)
        k = K.permute_dimensions(K.dot(key, self.Wk), (0,2,1))
        # l1*l2
        A = K.batch_dot(q,k)

        A = mask*A + (1-mask)*mask_value
        # b*l1*l2*1
        A = K.expand_dims(A, axis=-1)

        
        # Mask for attention dropout.
        def dropped_A():
            dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
            return A*dp_mask + (1-dp_mask)*mask_value
        A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))


        A = K.softmax(A, axis=-2)

        # b*1*l2*d
        value = K.expand_dims(value, axis=-3)
        
        # b*l1*l2*d
        value = A*value

        # b*l1*d
        value = K.sum(value, axis=-2)
        
        return value
        
    def compute_output_shape(self, input_shape):
        return input_shape


class ManualSpecialTransformer(Layer):
    
    def __init__(self, h, dk=None, dv=None, dff=None, dropout=0):
        self.h = h # times, values and varis
        self.dk, self.dv, self.dff, self.dropout = dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(ManualSpecialTransformer, self).__init__()

    def build(self, input_shape):
        d = input_shape[0].as_list()[-1] 
        
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.h, d, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.h, d, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.h, d, self.dv), name='Wv',
                                 initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.dv*self.h, d), name='Wo',
                                 initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(d, self.dff), name='W1',
                                 initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.dff), name='b1',
                                 initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.dff, d), name='W2',
                                 initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(d), name='b2',
                                 initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2,), name='gamma',
                                 initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2,), name='beta',
                                 initializer='zeros', trainable=True)
        super(ManualSpecialTransformer, self).build(input_shape)
        

    def call(self, inputs, mask, mask_value=-1e30):
        times, values, varis = inputs
        mask = K.expand_dims(mask, axis=-2)
        # MHA
        mha_ops = []

        query = [
            (0,2), (1,2), (0,2), (0,1), (1,2), (0,2)
        ]
        key = [
            (0,2), (1,2), (1,2), (0,1), (0,1), (0,1)
        ]
        value = times+varis+values
        for j in range(self.h):
            
            q = K.dot(inputs[query[j][0]] + inputs[query[j][1]], self.Wq[j,:,:])
            k = K.permute_dimensions(K.dot(inputs[key[j][0]] + inputs[key[j][1]], self.Wk[j,:,:]), (0,2,1))
            v = K.dot(value, self.Wv[j,:,:])
            A = K.batch_dot(q,k)

            # Mask unobserved steps.
            A = mask*A + (1-mask)*mask_value
            # Mask for attention dropout.
            def dropped_A():
                dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                return A*dp_mask + (1-dp_mask)*mask_value
            A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
            A = K.softmax(A, axis=-1)
            mha_ops.append(K.batch_dot(A,v))
        conc = K.concatenate(mha_ops, axis=-1)   
        proj = K.dot(conc , self.Wo)
        # Dropout.
        proj = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),\
                                   lambda: array_ops.identity(proj))
        # Add & LN
        x = value+proj
        mean = K.mean(x, axis=-1, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        x = (x - mean) / std
        x = x*self.gamma[0] + self.beta[0]
        # FFN
        ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1), self.b1)), 
                       self.W2), self.b2)
        # Dropout.
        ffn_op = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),\
                                   lambda: array_ops.identity(ffn_op))
        # Add & LN
        x = x+ffn_op
        mean = K.mean(x, axis=-1, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        x = (x - mean) / std
        x = x*self.gamma[1] + self.beta[1]   
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape


class ImputedTransformer(Layer):
    
    def __init__(self, N=2, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(ImputedTransformer, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.N, self.h, d+1, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.N, self.h, d+1, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.N, self.h, d+1, self.dv), name='Wv',
                                 initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.N, self.dv*self.h, d), name='Wo',
                                 initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(self.N, d, self.dff), name='W1',
                                 initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.N, self.dff), name='b1',
                                 initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.N, self.dff, d), name='W2',
                                 initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(self.N, d), name='b2',
                                 initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2*self.N,), name='gamma',
                                 initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2*self.N,), name='beta',
                                 initializer='zeros', trainable=True)
        super(ImputedTransformer, self).build(input_shape)
        
    def call(self, x, mask, imputed_mask, mask_value=-1e30):
        mask = K.expand_dims(mask, axis=-2)
        imputed_mask = K.expand_dims(imputed_mask, axis=-1)
        for i in range(self.N):
            # MHA
            mha_ops = []
            x_imputed_mask = K.concatenate([x, imputed_mask], axis=-1)
            for j in range(self.h):
                q = K.dot(x_imputed_mask, self.Wq[i,j,:,:])
                k = K.permute_dimensions(K.dot(x_imputed_mask, self.Wk[i,j,:,:]), (0,2,1))
                v = K.dot(x_imputed_mask, self.Wv[i,j,:,:])
                A = K.batch_dot(q,k)
                # Mask unobserved steps.
                A = mask*A + (1-mask)*mask_value
                # Mask for attention dropout.
                def dropped_A():
                    dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                    return A*dp_mask + (1-dp_mask)*mask_value
                A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
                A = K.softmax(A, axis=-1)
                mha_ops.append(K.batch_dot(A,v))
            conc = K.concatenate(mha_ops, axis=-1)
            proj = K.dot(conc , self.Wo[i,:,:])
            # Dropout.
            proj = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),\
                                       lambda: array_ops.identity(proj))
            # Add & LN
            x = x+proj
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i] + self.beta[2*i]
            # FFN
            ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1[i,:,:]), self.b1[i,:])), 
                           self.W2[i,:,:]), self.b2[i,:,])
            # Dropout.
            ffn_op = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),\
                                       lambda: array_ops.identity(ffn_op))
            # Add & LN
            x = x+ffn_op
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i+1] + self.beta[2*i+1]            
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape


class ImputedMultiTimeAttentionV1(Layer):

    def __init__(self, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.h, self.dk, self.dv, self.dff, self.dropout = h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(ImputedMultiTimeAttentionV1, self).__init__()

    def build(self, input_shape):
        d = input_shape[0].as_list()[-1]

        num_features = input_shape[2].as_list()[-1]
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.h, d, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.h, d, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        # self.Wv = self.add_weight(shape=(self.h, num_features, num_features), name='Wv',
        #                          initializer='glorot_uniform', trainable=True)
        self.Wc = self.add_weight(shape=(self.h, d), name='Wc',
                                 initializer='glorot_uniform', trainable=True)
        self.bc = self.add_weight(shape=(d), name='bc',
                                 initializer='zeros', trainable=True)
        self.Wo_1 = self.add_weight(shape=(num_features, int(np.sqrt(num_features))), name='Wo_1',
                                 initializer='glorot_uniform', trainable=True)
        self.bo_1 = self.add_weight(shape=(int(np.sqrt(num_features))), name='bo_1',
                                 initializer='zeros', trainable=True)
        self.Wo_2 = self.add_weight(shape=(int(np.sqrt(num_features)), 1), name='Wo_2',
                                 initializer='glorot_uniform', trainable=True)
        self.bo_2 = self.add_weight(shape=(d,), name='bo_2',
                                 initializer='zeros', trainable=True)
        super(ImputedMultiTimeAttentionV1, self).build(input_shape)
        
    def call(self, inputs, mask, mask_value=-1e30):
        query, key, value = inputs
        mask = K.expand_dims(mask, axis=-3) # b, 1, time_key, features
        # MHA
        mha_ops = []
        for j in range(self.h):
            q = K.dot(query, self.Wq[j,:,:]) # b, time_query, dk
            k = K.permute_dimensions(K.dot(key, self.Wk[j,:,:]), (0,2,1)) # b,time_key, dk
            # v = K.dot(value, self.Wv[j,:,:]) # b, time_key, features
            A = K.batch_dot(q,k) # b,time_query,time_key

            # Mask unobserved steps.
            A = K.expand_dims(A, axis=-1) # b, time_query, time_key, 1
            A = mask*A + (1-mask)*mask_value # b, time_query, time_key, features
            # Mask for attention dropout.
            def dropped_A():
                dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                return A*dp_mask + (1-dp_mask)*mask_value
            
            A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
            A = K.softmax(A, axis=-2) #Softmax at the time_key dim 

            v = K.expand_dims(value, axis=-3) # b, 1, time_key, features
            mha_ops.append(K.expand_dims(K.sum(A*v, axis=-2), axis=-1)) # b,time_query, time_key, features -> # b, time_query, features, 1

        
        latent_cve = K.concatenate(mha_ops, axis=-1)  # b,time_query, features, num_heads
        print(f'latent_cve: {latent_cve.get_shape()}')
        latent_cve = K.tanh(K.bias_add(K.dot(K.permute_dimensions(latent_cve, (0,2,1,3)), self.Wc), self.bc)) # b, features, time_query, num_heads -> b, features, time_query, d
        print(f'latent_cve: {latent_cve.get_shape()}')
        x = K.tanh(K.bias_add(K.dot((K.permute_dimensions(latent_cve, (0,2,3,1))), self.Wo_1), self.bo_1)) # b,time_query, d, feature -> b,time_query, d, sqrt(num_features)
        print(f'x: {x.get_shape()}')
        x = K.bias_add(K.squeeze(K.dot(x, self.Wo_2), axis=-1), self.bo_2) # b, time_query, d, 1 -> b, time_query, d
        print(f'x: {x.get_shape()}')
        return x


class ImputedMultiTimeAttentionV2(Layer):

    def __init__(self, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.h, self.dk, self.dv, self.dff, self.dropout = h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(ImputedMultiTimeAttentionV2, self).__init__()

    def build(self, input_shape):
        d = input_shape[0].as_list()[-1]

        num_features = input_shape[2].as_list()[-1]
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.h, d, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.h, d, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.h, num_features, num_features), name='Wv',
                                 initializer='glorot_uniform', trainable=True)
        self.Wc = self.add_weight(shape=(num_features, self.h, 1), name='Wc',
                                 initializer='glorot_uniform', trainable=True)
        self.bc = self.add_weight(shape=(num_features,1), name='bc',
                                 initializer='zeros', trainable=True)
        super(ImputedMultiTimeAttentionV2, self).build(input_shape)
        
    def call(self, inputs, mask, mask_value=-1e30):
        query, key, value = inputs
        mask = K.expand_dims(mask, axis=-3) # b, 1, time_key, features
        # MHA
        mha_ops = []
        for j in range(self.h):
            q = K.dot(query, self.Wq[j,:,:]) # b, time_query, dk
            k = K.permute_dimensions(K.dot(key, self.Wk[j,:,:]), (0,2,1)) # b,time_key, dk
            v = K.dot(value, self.Wv[j,:,:]) # b, time_key, features
            A = K.batch_dot(q,k) # b,time_query,time_key

            # Mask unobserved steps.
            A = K.expand_dims(A, axis=-1) # b, time_query, time_key, 1
            A = mask*A + (1-mask)*mask_value # b, time_query, time_key, features
            # Mask for attention dropout.
            def dropped_A():
                dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                return A*dp_mask + (1-dp_mask)*mask_value
            
            A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
            A = K.softmax(A, axis=-2) #Softmax at the time_key dim 

            v = K.expand_dims(v, axis=-3) # b, 1, time_key, features
            mha_ops.append(K.expand_dims(K.sum(A*v, axis=-2), axis=-1)) # b,time_query, time_key, features -> # b, time_query, features, 1

        
        latent_cve = K.concatenate(mha_ops, axis=-1)  # b,time_query, features, num_heads
        
        features = []
        for j in range(v.shape.as_list()[-1]):

            cur_feature = K.tanh(K.bias_add(K.dot(latent_cve[:,:,j,:], self.Wc[j,:,:]), self.bc[j,:])) # b, time_query, num_heads -> b, time_query, 1
            features.append(cur_feature) # (b,time_query, 1) * features
        
        x = K.concatenate(features, axis=-1) # b,time_query,features
        
        return x


class Time2Vec(Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(Time2Vec, self).__init__()
        
    def build(self, input_shape): 
        self.Wp = self.add_weight(name='T2V_Wp',
                            shape=(1, self.output_dim-1),
                            initializer='glorot_uniform',
                            trainable=True)
        self.bp = self.add_weight(name='T2V_b1',
                            shape=(self.output_dim-1,),
                            initializer='zeros',
                            trainable=True)
        self.Wl = self.add_weight(name='T2V_Wl',
                            shape=(1, 1),
                            initializer='glorot_uniform',
                            trainable=True)
        self.bl = self.add_weight(name='T2V_bl',
                            shape=(1,),
                            initializer='zeros',
                            trainable=True)
        super(Time2Vec, self).build(input_shape)
        
    def call(self, x):
        x = K.expand_dims(x, axis=-1)
        periodic_q = K.sin(K.bias_add(K.dot(x, self.Wp), self.bp))
        linear_q = K.bias_add(K.dot(x, self.Wl), self.bl)

        time_q = K.concatenate([periodic_q, linear_q], axis=-1)
        return time_q
        # time_query, time_key = x
        # time_query = K.expand_dims(time_query, axis=-1)
        # time_key = K.expand_dims(time_key, axis=-1)
        
        # periodic_q = K.sin(K.bias_add(K.dot(time_query, self.Wp), self.bp))
        # linear_q = K.bias_add(K.dot(time_query, self.Wl), self.bl)

        # time_q = K.concatenate([periodic_q, linear_q], axis=-1)
        
        # periodic_k = K.sin(K.bias_add(K.dot(time_key, self.Wp), self.bp))
        # linear_k = K.bias_add(K.dot(time_key, self.Wl), self.bl)

        # time_k = K.concatenate([periodic_k, linear_k], axis=-1)

        # return time_q, time_k

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)