import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, values, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, heads_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        # n: batch_size, q:query_len, h:heads, d:heads_dim, k:key_len
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum('nqhd, nkhd->nhqk', [query, keys])

        # query_len is target source sentence, key_len is source sentence
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # attention.shape: (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out shape: (N, query_len, heads, head_dim, )
        out = torch.einsum('nhql, nlhd->nqhd', [attention, values])

        # out.shape: (N, query_len, heads*heads_dim)
        # flatten the last two dimensions
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # last fc layer
        # out.shape: (N, query_len, heads*heads_dim)
        out = self.fc_out(out)

        return out
