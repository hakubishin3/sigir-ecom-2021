import torch.nn as nn


class EncoderEmbeddings(nn.Module):
    def __init__(self, encoder_params: dict) -> None:
        super().__init__()
        self.encoder_params = encoder_params

        self.id_embeddings = nn.Embedding(
            encoder_params["vocab_size"],
            encoder_params["embedding_size"],
            padding_idx=encoder_params["pad_token_id"],
        )

        self.linear_embed = nn.Linear(
            encoder_params["embedding_size"],
            encoder_params["hidden_size"],
        )
        self.layer_norm = nn.LayerNorm(
            encoder_params["hidden_size"],
            eps=encoder_params["layer_norm_eps"],
            )
        self.dropout = nn.Dropout(
            encoder_params["hidden_dropout_prob"]
        )

    def forward(self,
                input_ids=None,
                server_timestamp_epoch_sec=None,
                inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.id_embeddings(input_ids)

        #category_embeddings = self.category_embeddings(category_ids)
        #embeddings = torch.cat([inputs_embeds, category_embeddings], dim=-1)
        embeddings = inputs_embeds
        embeddings = self.linear_embed(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        encoder_params: dict,
        dropout: float = 0.3,
        hidden_size: int = 512,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_params = encoder_params
        self.encoder_params["vocab_size"] = num_labels + 1   # number of unique items + padding id

        self.embeddings = EncoderEmbeddings(self.encoder_params)
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_params["hidden_size"], nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(encoder_params["hidden_size"], encoder_params["hidden_size"]),
            nn.LayerNorm(encoder_params["hidden_size"]),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_params["hidden_size"], num_labels),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        server_timestamp_epoch_sec=None,
        encoder_outputs=None,
        inputs_embeds=None,
    ):
        if encoder_outputs is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                server_timestamp_epoch_sec=server_timestamp_epoch_sec,
                inputs_embeds=inputs_embeds,
            )
            encoder_outputs = self.encoder(
                embedding_output,
            )
        encoder_outputs = self.dropout(encoder_outputs)
        zuru = encoder_outputs[:, -1, :]
        logits = self.ffn(zuru)
        return logits
