import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion)
             for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):
            N, seq_length = x.shape
            positions = torch.arange(
                0,
                seq_length
            ).expand(
                N,
                seq_length
            ).to(
                self.device)

            x = self.dropout(
                self.word_embedding(x) + self.position_embedding(positions)
            )

            for layer in self.layers:
                x = layer(x, x, x, mask)

            return x
