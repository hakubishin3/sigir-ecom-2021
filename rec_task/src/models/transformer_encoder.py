import torch
import torch.nn as nn
from functools import partial


class EncoderEmbeddings(nn.Module):
    def __init__(self, encoder_params: dict) -> None:
        super().__init__()
        self.encoder_params = encoder_params

        make_embedding_func = partial(
            nn.Embedding,
            embedding_dim=encoder_params["embedding_size"],
            padding_idx=encoder_params["pad_token_id"],
        )
        self.id_embeddings = make_embedding_func(encoder_params["vocab_size"])
        self.elapsed_time_embeddings = make_embedding_func(encoder_params["max_elapsed_seconds"] + 1)
        self.product_action_embeddings = make_embedding_func(encoder_params["size_product_action"])
        self.hashed_url_embeddings = make_embedding_func(encoder_params["size_hashed_url"])
        self.price_bucket_embeddings = make_embedding_func(encoder_params["size_price_bucket"])
        self.number_of_category_hash_embeddings = make_embedding_func(encoder_params["size_number_of_category_hash"])
        self.category_hash_first_level_embeddings = make_embedding_func(encoder_params["size_category_hash_first_level"])
        self.category_hash_second_level_embeddings = make_embedding_func(encoder_params["size_category_hash_second_level"])
        self.category_hash_third_level_embeddings = make_embedding_func(encoder_params["size_category_hash_third_level"])

        self.linear_embed = nn.Linear(
            encoder_params["embedding_size"] * 9,
            encoder_params["hidden_size"],
        )
        self.layer_norm = nn.LayerNorm(
            encoder_params["hidden_size"],
            eps=encoder_params["layer_norm_eps"],
            )
        self.dropout = nn.Dropout(
            encoder_params["hidden_dropout_prob"]
        )

    def forward(
        self,
        input_ids=None,
        elapsed_time=None,
        product_action=None,
        hashed_url=None,
        price_bucket=None,
        number_of_category_hash=None,
        category_hash_first_level=None,
        category_hash_second_level=None,
        category_hash_third_level=None,
    ):
        inputs_embeds = self.id_embeddings(input_ids)

        # elapsed time as categorical embedding
        elapsed_time_cat = (elapsed_time.long() + 1).clamp(min=0, max=self.encoder_params["max_elapsed_seconds"])
        elapsed_time_embeds = self.elapsed_time_embeddings(elapsed_time_cat)

        other_embeddings = [
            self.product_action_embeddings(product_action),
            self.hashed_url_embeddings(hashed_url),
            self.price_bucket_embeddings(price_bucket),
            self.number_of_category_hash_embeddings(number_of_category_hash),
            self.category_hash_first_level_embeddings(category_hash_first_level),
            self.category_hash_second_level_embeddings(category_hash_second_level),
            self.category_hash_third_level_embeddings(category_hash_third_level),
        ]

        embeddings = torch.cat([inputs_embeds, elapsed_time_embeds] + other_embeddings, dim=-1)
        embeddings = self.linear_embed(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        encoder_params: dict,
        dropout: float = 0.3,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_params = encoder_params
        self.encoder_params["vocab_size"] = num_labels   # number of unique items + padding id

        self.embeddings = EncoderEmbeddings(self.encoder_params)
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_params["hidden_size"], nhead=encoder_params["nhead"])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_params["num_layers"])
        self.dropout = nn.Dropout(dropout)
        self.global_max_pooling_1d = GlobalMaxPooling1D()
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
        elapsed_time=None,
        product_action=None,
        hashed_url=None,
        price_bucket=None,
        number_of_category_hash=None,
        category_hash_first_level=None,
        category_hash_second_level=None,
        category_hash_third_level=None,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            elapsed_time=elapsed_time,
            product_action=product_action,
            hashed_url=hashed_url,
            price_bucket=price_bucket,
            number_of_category_hash=number_of_category_hash,
            category_hash_first_level=category_hash_first_level,
            category_hash_second_level=category_hash_second_level,
            category_hash_third_level=category_hash_third_level,
        )
        encoder_outputs = self.encoder(
            embedding_output,
        )
        encoder_outputs = self.dropout(encoder_outputs)
        pooling =self.global_max_pooling_1d(encoder_outputs)
        logits = self.ffn(pooling)
        return logits


class GlobalMaxPooling1D(nn.Module):
    """https://discuss.pytorch.org/t/equivalent-of-keras-globalmaxpooling1d/45770/4
    """
    def __init__(self, data_format='channels_last'):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values
