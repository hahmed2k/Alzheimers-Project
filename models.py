import torch.nn as nn
import torch

class AlzheimerMLP(nn.Module):
    def __init__(self, num_numerical, cat_sizes, embedding_dim=8, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, embedding_dim) for size in cat_sizes])
        input_dim = num_numerical + len(cat_sizes) * embedding_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, num_inputs, cat_inputs):
        embeds = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embeds + [num_inputs], dim=1)
        return self.fc(x)

class TabTransformer(nn.Module):
    def __init__(self, num_numerical, cat_sizes, emb_dim=32, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(size + 1, emb_dim) for size in cat_sizes])
        self.num_projection = nn.Linear(1, emb_dim)
        self.total_num_tokens = num_numerical + len(cat_sizes) + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embeddings = nn.Parameter(torch.randn(1, self.total_num_tokens, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(emb_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x_num, x_cat):
        num_emb = self.num_projection(x_num.unsqueeze(-1))
        cat_emb = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        feature_tokens = torch.cat([num_emb, cat_emb], dim=1)
        cls_tokens = self.cls_token.expand(feature_tokens.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1) + self.pos_embeddings
        out = self.transformer_encoder(tokens)
        return self.head(out[:, 0, :])
