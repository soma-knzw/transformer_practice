import torch.nn as nn

from self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # add skip connection, run through norm and funally dopout
        x = self.norm1(attention + query)
        x = self.dropout(x)

        # feed_forward layers
        forward = self.feed_forward(x)

        # skip connection, run through norm and dropout
        x = self.norm2(forward + x)
        x = self.dropout(x)

        return x
