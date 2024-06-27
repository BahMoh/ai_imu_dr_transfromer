import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

import numpy as np
import matplotlib.pyplot as plt

# We had two different MHA layers
# Encoder doesn't need a causal mask
# Decoder needs that

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, causal=False):
        super().__init__()

        self.d_k = d_k
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)

        # final linear layer
        self.fc = nn.Linear(d_k * n_heads  * 6, d_model)

        # causal mask
        # make it so that diagonal is zero too!!!
        # This way we don't need to shift the inputs to make targets
        self.causal = causal
        if causal:
            # max_len is the same as T
            cm = torch.tril(torch.ones(max_len, max_len))
            self.register_buffer(
                "causal_mask",
                cm.view(1, 1, max_len, max_len)
            )

        self.to(device)

    # q, k, v passed as the arguments of the below method are the sane as x input
    def forward(self, q, k, v, pad_mask=None):
        q = self.query(q)  # N * T * (h.d_k)
        k = self.key(k)    # N * T * (h.d_k)
        v = self.value(v)  # N * T * (h.d_k)

        # Which decoder output should pay attetntion to which encoder input??
        # This is related to wheer the decoder is connected to the encoder
        N = q.shape[0]
        T_output  = q.shape[1]
        T_input = k.shape[1]

        # Compute attention weights
        # (N, T, h.d_k) -> (N, T, h, d_k)
        # (N, T, h, d_k) -> (N, h, T, d_k)

        # q comes from  decoder
        q = q.view(N, T_output, self.n_heads, self.d_k).transpose(1,2)
        # k, v come from encoder. So they can have different sequence lengths
        k = k.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)
        v = v.view(N, T_input, self.n_heads, self.d_k).transpose(1,2)

        # Compute attention score
        # (N, h, T, d_k) x (N, h, d_k, T) -> (N, h, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(
                pad_mask[:, None, None, :] == 0, float("-inf"))
        # print(causal_mask.shape)
        if self.causal:
            print(T_output, T_input)
            attn_scores = attn_scores.masked_fill(
                # third index goes up to T_output, the fourth index goes to T_input
                self.causal_mask[:, :, T_output, :T_input] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim = -1)

        # compute attention-weighted values
        # (N, h, T, T) x (N, h, T, d_k) --> (N, h, T, d_k)
        A = attn_weights @ v

        # reshape it back before final linear layer
        A = A.transpose(1, 2) # (N, T, h, d_k)
        ################################################################# Attention ##################################
        A = A.contiguous().view(N, T_output, self.d_k * self.n_heads * 6) # (N, T, 6 * h*d_k)
        ##############################################################################################################
        # projection
        return self.fc(A) # (N, T, d_model)


# Transformer block in the encoder notebook
class EncoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # We don't need casual mask here
        self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        # self.mha.to(device)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear( d_model * 4, d_model),
            nn.Dropout(dropout_prob)
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, mask=None):
        mha_output = self.mha(x, x, x, mask)
        mha_output.to(device)
        x.to(device)
        x = self.ln1(x.view(mha_output.shape[0], mha_output.shape[1], mha_output.shape[2]) + mha_output)
        x = self.ln2(x + self.ann(x))
        x = self.dropout(x)
        return x

# Transformer block from decoder notebook
# Very differnet from the original notebook
class DecoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        # 2 mha's, one for causal self attention one for taking the input from encoder output
        # One is masked one is not!
        self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=True)
        self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        # self.mha1.to(device)
        # self.mha2.to(device)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob)
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        # self-attention on decoder input
        x = self.ln1(
            dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask))

        # multi-head attention including encoder output
        x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))

        x = self.ln3(x + self.ann(x))
        x = self.dropout(x)
        return x


# same as before
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # print(f"position {position.shape}")
        # print(f"exp_term {exp_term.shape}")
        # print(f"div_term {div_term.shape}")
        # print(f"pe {pe.shape}")
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.shape: N x T x D
        # print(f"x.shape {x.shape}, pe.shape {self.pe.shape}, 160 seq2seq")
        # print(f"self.pe[:, :x.size(1), :] {self.pe[:, :x.size(1), :].shape} 160 seq2seq")
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x )

# Comment out the things we don't need
class Encoder(nn.Module):
    def __init__(self,
                #  vocab_size,
                 # input_dim,
                 max_len,
                 d_k,
                 d_model,
                 n_heads,
                 n_layers,
                #  n_classes,
                 dropout_prob):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, d_model)

        # max_len is used in the PositionalEncoding class to create positional
        # encodings for sequences up to this maximum length. The positional
        # encoding tensor is of size (1, max_len, d_model), allowing the model
        # to handle sequences of any length up to max_len.
        self.input_embedding = nn.Linear(6, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformer_blocks = [
            EncoderBlock(
                d_k,
                d_model,
                n_heads,
                dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        # self.fc = nn.Linear(d_model, n_classes)
        # self.to(device)
    def forward(self, x, mask=None):
        # print(x.shape, "x raw")
        x = x.view(x.shape[0], x.shape[1], -1)
        # print(x.dtype)
        x = x.float()
        # x = self.embedding(x)
        # print(x.shape, "seq_to_seq_transformer.py, line 201, in forward")
        x = x.transpose(1,2)
        x = x.to(device)
        # print(x.device)
        # print(self.input_embedding.weight.device)
        x = self.input_embedding(x)
        # print(x.shape, "after input embedding")
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x.to(device), mask.to(device))

        # many-to-one (x has the shep N x T x D)
        # This is optional, assuming we have text classification
        # And only need to keep one of the hidden vectors
        # x = x[:, 0, :]

        x = self.ln(x)
        # x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                # vocab_size,
                # input_dim,
                max_len,
                d_k,
                d_model,
                n_heads,
                n_layers,
                n_output,
                dropout_prob):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, d_model)

        #input embedding
        self.input_embedding = nn.Linear(6, d_model)
        self.input_embedding.to(device)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformer_blocks = [
            DecoderBlock(
                d_k,
                d_model,
                n_heads,
                max_len,
                dropout_prob) for _ in range(n_layers)]

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_output)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        # x = self.embedding(dec_input)
        # dec_input = dec_input.view(dec_input.shape[0], dec_input.shape[1], 1)
        dec_input = dec_input.float()
        dec_input = dec_input.to(device)


        # print(dec_input.shape)
        dec_input = dec_input.transpose(1,2)
        x = self.input_embedding(dec_input)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(enc_output, x, enc_mask, dec_mask)
        x = self.ln(x)
        x = self.fc(x) # Many2Many task
        return x

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(enc_output, dec_input, enc_mask, dec_mask)
        return dec_output



import sys
if __name__ == "__main__":
    encoder = Encoder(max_len=512,
                 d_k=16,
                 d_model=64,
                 n_heads=4,
                 n_layers=2,
                 dropout_prob=0.1)

    decoder = Decoder(max_len=512,
                    d_k=16,
                    d_model=64,
                    n_heads=4,
                    n_layers=2,
                    n_output = 2,
                    dropout_prob=0.1)

    transformer = Transformer(encoder, decoder)
    model_size = sum(param.numel() for param in transformer.parameters())
    print(f"Model size: {model_size / 1e6:.2f} Mb")


