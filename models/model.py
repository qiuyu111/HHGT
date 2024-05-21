import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_softmax, scatter_add
from torch.utils.checkpoint import checkpoint
from torch_sparse import spmm
import math
import torch.nn.functional as F
import torch.nn as nn

def init_params(module, L):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(L))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class HierarchicalTransformer(nn.Module):
    def __init__(
        self,
        type,
        hops, 
        input_dim, 
        L_hop,
        L_type,
        num_heads=8,
        d_model=512,
        dropout_rate=0.0,
        attention_dropout_rate=0.1,
        use_gradient_checkpointing=False
    ):
        super().__init__()

        self.L_hop = L_hop
        self.L_type = L_type
        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.d_model = d_model
        self.ffn_dim = 2 * d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.L = L_hop + L_type
        self.type = type

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.d_model)

        self.type_transformer = TypeTransformer(
            type = self.type,
            seq = self.seq_len,
            d_model=self.d_model,
            ffn_dim=self.ffn_dim,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
            L_type=self.L_type,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        self.hop_transformer = HopTransformer(
            seq = self.seq_len,
            d_model=self.d_model,
            ffn_dim=self.ffn_dim,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
            L_hop=self.L_hop,
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        self.apply(lambda module: init_params(module, self.L))

    def forward(self, processed_nodes_features):
        '''linear projection'''
        tensor = self.att_embeddings_nope(processed_nodes_features)
        # Type Transformer
        type_transformer_output = self.type_transformer(tensor)
        # Hop Transformer with the output of Type Transformer as input
        hop_transformer_output = self.hop_transformer(type_transformer_output)
        return hop_transformer_output

class TypeTransformer(nn.Module):
    def __init__(self, type, seq, d_model, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, L_type, use_gradient_checkpointing):
        super(TypeTransformer, self).__init__()

        encoders = [TransformerLayer_type(d_model, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(L_type)]
        self.layers = nn.ModuleList(encoders) 
        self.final_ln = nn.LayerNorm(d_model) 
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.seq = seq
        self.type = type
        self.linear_layer = nn.Linear(self.type, 1)
        self.mlp = nn.Sequential(
            nn.Linear(self.type * d_model, 256),  
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, x):
        for enc_layer in self.layers:
            if self.use_gradient_checkpointing:
                x = checkpoint(enc_layer, x)
            else:
                x = enc_layer(x)

        '''LayerNorm'''
        output_type = self.final_ln(x) 

        '''attention mechanism'''
        node_self = output_type[:,:,0,:,:]
        node_self_concate = node_self.view(node_self.size(0), node_self.size(1), -1)
        node_self_concate = self.mlp(node_self_concate)
        node_self_agg = node_self_concate.view(node_self.size(0), node_self.size(1), 1, node_self.size(-1))
        node_self_agg_flat = node_self_agg.view(node_self_agg.size(0)*node_self_agg.size(1),1,node_self_agg.size(3))

        type_all = []
        for k in range(2, self.seq + 1):
            sub_tensor = output_type[:, :, k - 1, :, :]
            sub_tensor_trans = torch.transpose(sub_tensor,2,3)
            sub_tensor_trans_flat = sub_tensor_trans.view(sub_tensor_trans.size(0)*sub_tensor_trans.size(1),sub_tensor_trans.size(2),sub_tensor_trans.size(3))
            atten_flat = torch.bmm(node_self_agg_flat,sub_tensor_trans_flat)
            sub_tensor_flat = torch.transpose(sub_tensor_trans_flat,1,2)
            type_tensor = torch.bmm(atten_flat,sub_tensor_flat)
            type_all.append(type_tensor)
        concatenated_tensor = torch.cat(type_all, dim=1)
        concatenated_tensor = concatenated_tensor.view(node_self_agg.size(0),node_self_agg.size(1),concatenated_tensor.size(-2),concatenated_tensor.size(-1))
        type_output = torch.cat([node_self_agg,concatenated_tensor], dim=2)
        return type_output

class HopTransformer(nn.Module):
    def __init__(self, seq, d_model, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, L_hop, use_gradient_checkpointing):
        super(HopTransformer, self).__init__()

        encoders = [TransformerLayer_hop(d_model, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(L_hop)]
        self.layers = nn.ModuleList(encoders) 
        self.final_ln = nn.LayerNorm(d_model) 
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.seq_len = seq
        self.attn_layer = nn.Linear(2 * d_model, 1)

    def forward(self, x):
        for enc_layer in self.layers:
            if self.use_gradient_checkpointing:
                x = checkpoint(enc_layer, x)
            else:
                x = enc_layer(x)
        output = self.final_ln(x)
        '''attention mechanism'''
        target = output[:,:,0,:].unsqueeze(2).repeat(1,1,self.seq_len-1,1) 
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=2)
        node_tensor = split_tensor[0] 
        neighbor_tensor = split_tensor[1] 
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=3)) 
        layer_atten = F.softmax(layer_atten, dim=2) 
        neighbor_tensor = neighbor_tensor * layer_atten 
        neighbor_tensor = torch.sum(neighbor_tensor, dim=2, keepdim=True) 
        output = (node_tensor + neighbor_tensor).squeeze() 
        return output

class TransformerLayer_type(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(TransformerLayer_type, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention_type(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, attn_bias=None):
        x_set = []

        for i in range(x.shape[0]):
            node_each = x[i, :, :, :, :]

            # Multi-Head Self Attention (MSA)
            y = self.self_attention_norm(node_each)
            y = self.self_attention(y, y, y, attn_bias)
            y = self.self_attention_dropout(y)
            z = node_each + y

            # Feed Forward Network (FFN)
            y = self.ffn_norm(z)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            z = z + y

            x_set.append(z)

        x_set = torch.stack(x_set, dim=0)
        
        return x_set
    

class TransformerLayer_hop(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(TransformerLayer_hop, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention_hop(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, attn_bias=None):
        x_set = []

        for i in range(x.shape[0]):
            node_each = x[i, :, :, :]

            # Multi-Head Self Attention (MSA)
            y = self.self_attention_norm(node_each)
            y = self.self_attention(y, y, y, attn_bias)
            y = self.self_attention_dropout(y)
            z = node_each + y

            # Feed Forward Network (FFN)
            y = self.ffn_norm(z)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            z = z + y

            x_set.append(z)

        x_set = torch.stack(x_set, dim=0)
        
        return x_set


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x) 
        x = self.gelu(x) 
        x = self.layer2(x) 
        return x


class MultiHeadAttention_type(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention_type, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size() 
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        type_size = q.size(2)

        '''Scaled Dot-Product Attention
        head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)'''
        q = self.linear_q(q).view(batch_size, -1, type_size, self.num_heads, d_k) 
        k = self.linear_k(k).view(batch_size, -1, type_size, self.num_heads, d_k) 
        v = self.linear_v(v).view(batch_size, -1, type_size, self.num_heads, d_v) 

        q = q.transpose(2, 3)                 
        v = v.transpose(2, 3)           
        k = k.transpose(2, 3).transpose(3, 4)  
        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=4)
        x = self.att_dropout(x)
        x = x.matmul(v)  

        x = x.transpose(2, 3).contiguous() 
        x = x.view(batch_size, -1, type_size, self.num_heads * d_v) 

        x = self.output_layer(x) 
        assert x.size() == orig_q_size
        return x


class MultiHeadAttention_hop(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention_hop, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size() 
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        '''Scaled Dot-Product Attention
        head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)'''
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k) 
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k) 
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v) 

        q = q.transpose(1, 2)                  
        v = v.transpose(1, 2)                  
        k = k.transpose(1, 2).transpose(2, 3)  

        q = q * self.scale
        x = torch.matmul(q, k)
        
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  
        x = x.transpose(1, 2).contiguous() 
        x = x.view(batch_size, -1, self.num_heads * d_v) 
        x = self.output_layer(x) 
        assert x.size() == orig_q_size
        return x

