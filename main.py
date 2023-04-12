# -*- coding: utf-8 -*-
"""
# @File        : main.py
# @Time        : 2023/4/6 9:48
# @Author      : lhc
# @Description :
"""
import torch
from torch import nn
from dataLoader import *
import torch.optim as optim
from model import Transformer

def train():
    # 定义模型参数
    d_model = 512
    d_att = 64
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    input_vocab_size = 10000
    target_vocab_size = 10000

    # loadData
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    # 创建 Transformer 模型实例
    model = Transformer(d_model=d_model, d_att=d_att, d_ff=d_ff, n_heads=n_heads, src_vocab_size=10000,
                        tgt_vocab_size=10000, n_layers=n_layers).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(1000):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # inputs: enc_inputs->[2,5]; ec_inputs->[2,6]
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.8f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train()

