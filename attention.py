# -*- coding: utf-8 -*-
"""
# @File        : attention.py
# @Time        : 2023/4/6 9:29
# @Author      : lhc
# @Description : 
"""
import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_att):
        super(ScaledDotProductAttention, self).__init__()
        self.d_att = d_att

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, len_q, d_k]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v(=len_k), d_v]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]
        :return:
        """
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_att)
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_att, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_att = d_att
        self.n_heads = n_heads
        self.W_Q = nn.Linear(self.d_model, self.d_att * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_att * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_att * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_att, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v(=len_k), d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_att).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_att).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_att).transpose(1, 2)

        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_att)(Q, K, V, attn_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_att)
        # output: [batch_size, len_q, d_model]
        output = self.fc(context)

        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn
