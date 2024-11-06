import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from acds.archetypes.esn import ReservoirCell 

class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0

class GroupLinearLayer(nn.Module):
    def __init__(self, din, dout, num_blocks, normalize=False):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

        self.normalize = normalize
        if self.normalize:
            print('using batch norm')
            self.batch_norm = nn.BatchNorm1d(dout)

    def forward(self, x):
        perm_x = x.permute(1, 0 ,2)
        out = torch.bmm(perm_x, self.w).permute(1, 0, 2)

        if self.normalize:
            norm_out = self.batch_norm(out.reshape(-1, out.shape[2]))
            out = norm_out.reshape(out.shape)

        return out

class RIM(nn.Module):
    def __init__(
        self, input_size, device,
        num_units, active_units,
        use_input_attention, num_input_heads, key_input_size, query_input_size, value_input_size, input_dropout,
        use_comm_attention, num_comm_heads, key_comm_size, query_comm_size, value_comm_size, comm_dropout, alpha, use_value_comm,
        hidden_size, input_scaling, spectral_radius, leaky
    ):

        super().__init__()

        if value_comm_size != hidden_size:
            value_comm_size = hidden_size

        self.num_units = num_units
        self.active_units = active_units
        self.hidden_size = hidden_size
        self.key_input_size = key_input_size
        self.num_comm_heads = num_comm_heads
        self.key_comm_size = key_comm_size
        self.value_comm_size = value_comm_size
        self.query_comm_size = query_comm_size
        self.alpha = alpha
        self.use_value_comm = use_value_comm # if False, remove value_comm matrix
        self.use_input_attention = use_input_attention
        self.use_comm_attention = use_comm_attention
        self.device = device

        # define the modules
        connectivity = int(0.7 * hidden_size); print(f'Hard-Coded! Connectivity = {connectivity}')
        self.rnn = nn.ModuleList([
            ReservoirCell(value_input_size, units=hidden_size,
            input_scaling=input_scaling, spectral_radius=spectral_radius, leaky=leaky,
            connectivity_input=connectivity, connectivity_recurrent=connectivity)
            for _ in range(num_units)
        ])

        # define input attention layers
        self.input_key = nn.Linear(input_size, num_input_heads * key_input_size) #one per element
        self.input_value = nn.Linear(input_size, num_input_heads * value_input_size) #one per element
        self.input_query = GroupLinearLayer(hidden_size, num_input_heads * query_input_size, num_units) #one per module
        self.input_attention_dropout = nn.Dropout(p = input_dropout)

        # define communication attention layers
        self.comm_key = GroupLinearLayer(hidden_size, num_comm_heads * key_comm_size, num_units)
        if self.use_value_comm: self.comm_value = GroupLinearLayer(hidden_size, num_comm_heads * value_comm_size, num_units, normalize=True)
        self.comm_query = GroupLinearLayer(hidden_size, num_comm_heads * query_comm_size, num_units)
        self.communication_attention_dropout = nn.Dropout(p = comm_dropout)
        # needed to get the right shapes
        self.comm_output = GroupLinearLayer(num_comm_heads * value_comm_size, value_comm_size, num_units)

        # define readout
        # two layers for stability reasons
        self.readout = nn.Sequential(
            nn.Linear(hidden_size * num_units, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

        print('Input Attention: ', end=''); print('TRUE' if use_input_attention else 'FALSE')
        print('Communication Attention: ', end=''); print('TRUE' if use_comm_attention else 'FALSE')
        print('Unitary value_comm Matrix: ', end=''); print('TRUE' if not use_value_comm else 'FALSE')

    def forward(self, x):
        # initialize hidden state for each batch
        hs = torch.ones(x.shape[0], self.num_units, self.hidden_size) * 0.001
        hs = hs.to(self.device)

        x = x.float().squeeze().to(self.device)
        x = torch.split(x, 1, 1)

        hs_history = []
        for i, xt in enumerate(x):
            hs = self.single_timestep_forward(xt, hs)

            hs_history.append(hs.detach().cpu())
            try:
                assert not torch.isnan(hs).any().item(), f'hs NaN after {i} timesteps'
            except:
                import matplotlib.pyplot as plt

                hs_history = torch.stack(hs_history).squeeze()
                hs_history = torch.mean(hs_history, dim=1)
                hs_history = torch.mean(hs_history, dim=-1)
                plt.plot(hs_history)
                plt.savefig('hs_history')

                raise

        hs = hs.view(hs.shape[0], -1) # concatenate hs of all units
        y_pred = self.readout(hs)
        return y_pred

    def single_timestep_forward(self, x, hs):
        h_old = hs

        # concat null input
        x = x.unsqueeze(1)
        null_input = torch.zeros(x.shape[0], 1, x.shape[2]).to(self.device)
        x = torch.cat([x, null_input], dim=1)

        # compute input attention
        if self.use_input_attention:
            att_x, mask = self.input_attention(x, hs)
        else:
            att_x = x.repeat(self.num_units, 1)
            mask = torch.ones(x.shape[0], self.num_units).to(self.device)

        # listify the input and the hs
        # to pass the correct data to the corresponding module
        att_x = torch.split(att_x, 1, 1)
        att_x = [i.squeeze(1) for i in att_x]
        hs = torch.split(hs, 1, 1)
        hs = [shs.squeeze(1) for shs in hs]

        # forward pass on all ESNs modules
        for i in range(self.num_units):
            _, hs[i] = self.rnn[i](att_x[i], hs[i])
        hs = torch.stack(hs, dim = 1)

        # block gradient through inactive modules
        mask = mask.unsqueeze(2)
        h_new = blocked_grad.apply(hs, mask)

        # compute communication attention
        if self.use_comm_attention:
            h_new = self.communication_attention(h_new, mask, self.alpha)

        # update the hidden state for active units
        # while keeping the old hidden state for inactive ones
        hs = mask * h_new + (1 - mask) * h_old

        return hs

    def input_attention(self, x, hs):
        k = self.input_key(x)
        v = self.input_value(x)
        q = self.input_query(hs)

        # standard attention computation
        attention_scores = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(self.key_input_size))
        attention_scores = F.softmax(attention_scores, dim = -1)

        # compute mask over the input
        # based on top-k scoring modules
        # we actually select the modules with lowest score on null input
        null_attention_scores = attention_scores[:, :, 1] # using only the att_scores on null input
        smallest_k = torch.topk(null_attention_scores, self.active_units, dim = 1, largest = False) # k smallest elements

        mask = torch.zeros(x.shape[0], self.num_units).to(self.device)
        mask.scatter_(1, smallest_k.indices, 1)

        '''
        mask_ = torch.zeros(x.shape[0], self.num_units).to(self.device)
        for i, row in enumerate(smallest_k.indices):
            for elem in row:
                mask_[i][elem] = 1
        assert (mask == mask_).all()
        '''

        attention_scores = self.input_attention_dropout(attention_scores)
        attention_scores = torch.matmul(attention_scores, v)

        masked_input = attention_scores * mask.unsqueeze(2)
        return masked_input, mask

    def communication_attention(self, hs, mask, alpha):
        k = self.view_multihead(self.comm_key(hs), self.num_comm_heads, self.key_comm_size)
        q = self.view_multihead(self.comm_query(hs), self.num_comm_heads, self.query_comm_size)

        if self.use_value_comm:
            v = self.view_multihead(self.comm_value(hs), self.num_comm_heads, self.value_comm_size)

        # remove value_comm matrix
        # i.e. it is unitary matrix and it's not updated during training
        else:
            v = [hs for _ in range(self.num_comm_heads)]
            v = torch.stack(v, dim=1)

        # replicate mask for each attention head
        mask = [mask for _ in range(self.num_comm_heads)]
        mask = torch.stack(mask, dim=1)
        # mask to get only the active modules
        attention_scores = torch.matmul(q, k.transpose(-1, -2) / math.sqrt(self.key_comm_size))
        attention_scores = attention_scores * mask

        # standard attention with dropout
        attention_scores = F.softmax(attention_scores, dim = -1)
        attention_scores = self.communication_attention_dropout(attention_scores)
        attention_scores = torch.matmul(attention_scores, v)

        # att_scores now is of shape (batch_size, num_heads, num_units, value_comm_size)
        # to sum it with h_old, we need it to become (batch_size, num_units, hidden_size)
        attention_scores = attention_scores.reshape(attention_scores.shape[0], self.num_units, -1)
        attention_scores = self.comm_output(attention_scores)

        # compute the update of the hidden state
        # new hidden state is the weighted sum of attention_scores and old hs
        # or the unweighted sum
        if alpha:
            hs_update = alpha * attention_scores + (1 - alpha) * hs
        else:
            hs_update = attention_scores + hs

        return hs_update

    def view_multihead(self, tensor, num_att_heads, att_size):
        '''
        Returns a different view of an attention tensor, dividing into the different attention heads.
        tensor is of shape (batch_size, num_units, num_att_heads * att_size).
        output tensor is of shape (batch_size, num_att_heads, num_units, att_size)
        '''
        return tensor.reshape(tensor.shape[0], num_att_heads, -1, att_size)