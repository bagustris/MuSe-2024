# import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from config import ACTIVATION_FUNCTIONS, device

class RNN(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        n_layers=1,
        bi=True,
        dropout=0.2,
        n_to_1=False,
        rnn_type="gru",
        residual=False,
    ):
        super(RNN, self).__init__()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=d_in,
                hidden_size=d_out,
                bidirectional=bi,
                num_layers=n_layers,
                dropout=dropout,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=d_in,
                hidden_size=d_out,
                bidirectional=bi,
                num_layers=n_layers,
                dropout=dropout,
            )
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=d_in,
                hidden_size=d_out,
                bidirectional=bi,
                num_layers=n_layers,
                dropout=dropout,
            )
        else:
            raise ValueError("RNN type not supported")

        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1
        # self.attention = Attention(d_out * self.n_directions)
        # Add batch normalization
        self.bn = nn.BatchNorm1d(d_out * self.n_directions)

        # Add residual connections
        if residual and n_layers > 1:
            self.residual_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_out * self.n_directions, d_out * self.n_directions),
                        nn.BatchNorm1d(d_out * self.n_directions),
                    )
                    for _ in range(n_layers - 1)
                ]
            )
        else:
            self.residual_layers = None

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            # last_item_from_packed(rnn_enc[0], x_len)
            # last_seq_items = last_item_from_packed(rnn_enc[0], x_len)
            # attn_weights = self.attention(last_seq_items, rnn_enc[0].data)
            # context = attn_weights.bmm(rnn_enc[0].data.transpose(0, 1))
            # return context.squeeze(1)
            return last_item_from_packed(rnn_enc[0], x_len)


        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

             # Apply batch normalization
            x_out = x_out.permute(0, 2, 1)  # Reshape for batch normalization
            x_out = self.bn(x_out)
            x_out = x_out.permute(0, 2, 1)  # Reshape back to original shape
            
             # Apply residual connections
            if self.residual_layers is not None:
                x_residual = x_out
                for layer in self.residual_layers:
                    x_residual = layer(x_residual)
                    x_out = x_out + x_residual


        return x_out

#https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
def last_item_from_packed(packed, lengths):
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    )).to(device)
    sorted_lengths = lengths[packed.sorted_indices].to(device)
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0)).to(device)
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items

class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0, activation="relu"):
        super(OutLayer, self).__init__()
        if activation == "relu":
            self.activation = nn.ReLU(True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU(True)
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'rrelu':
            self.activation = nn.RReLU(True)
        elif activation == "mish":
            self.activation = nn.Mish(True)
        else:
            raise ValueError("Activation function not supported")
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), self.activation, nn.Dropout(dropout))
        # self.fc_2 = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        x = self.fc_1(x)
        # x = self.fc_2(x)
        y = self.fc_2(x)
        return y


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)


        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.encoder(x, x_len)
        y = self.out(x)
        activation = self.final_activation(y)
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1

