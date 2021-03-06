import numpy as np
import gensim
import torch
import torch.nn as nn
from functools import partial


class EncoderEmbeddings(nn.Module):
    def __init__(self, encoder_params: dict, preprocessor) -> None:
        super().__init__()
        self.encoder_params = encoder_params
        self.preprocessor = preprocessor

        w2v_model = gensim.models.KeyedVectors.load(
            "data/output/train_w2v/word2vec.model",
        )
        id_vec_list = []
        for label in range(encoder_params["vocab_size"]):
            if label in [0, 1]:
                # 0 is padding id, 1 is nan
                vec = np.random.normal(0.0, 0.01, size=encoder_params["embedding_size"])
            else:
                product_sku_hash = preprocessor.index_to_label_dict["product_sku_hash"][label]
                try:
                    vec = w2v_model.wv[product_sku_hash]
                except:
                    vec = np.random.normal(0.0, 0.01, size=encoder_params["embedding_size"])
            id_vec_list.append(vec)

        id_vectors = np.array(id_vec_list)
        print(encoder_params["vocab_size"])
        print(id_vectors.shape)
        id_weights = torch.FloatTensor(id_vectors)
        self.id_embeddings = nn.Embedding.from_pretrained(
            id_weights,
            freeze=False,
            padding_idx=encoder_params["pad_token_id"],
        )

        make_embedding_func = partial(
            nn.Embedding,
            embedding_dim=encoder_params["embedding_size"],
            padding_idx=encoder_params["pad_token_id"],
        )
        self.elapsed_time_embeddings = make_embedding_func(encoder_params["size_elapsed_time"])
        self.event_type_embeddings = make_embedding_func(encoder_params["size_event_type"])
        self.product_action_embeddings = make_embedding_func(encoder_params["size_product_action"])
        self.hashed_url_embeddings = make_embedding_func(encoder_params["size_hashed_url"])
        self.price_bucket_embeddings = make_embedding_func(encoder_params["size_price_bucket"])
        self.number_of_category_hash_embeddings = make_embedding_func(encoder_params["size_number_of_category_hash"])
        self.category_hash_first_level_embeddings = make_embedding_func(encoder_params["size_category_hash_first_level"])
        self.category_hash_second_level_embeddings = make_embedding_func(encoder_params["size_category_hash_second_level"])
        self.category_hash_third_level_embeddings = make_embedding_func(encoder_params["size_category_hash_third_level"])

        self.hour_embeddings = make_embedding_func(encoder_params["size_hour"])
        self.weekday_embeddings = make_embedding_func(encoder_params["size_weekday"])
        self.weekend_embeddings = make_embedding_func(encoder_params["size_weekend"])

        self.position_embeddings = nn.Embedding(
            encoder_params["window_size"],
            embedding_dim=encoder_params["hidden_size"],
        )

        self.linear_embed = nn.Linear(
            encoder_params["embedding_size"] * 13 + 50 * 3,
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
        event_type=None,
        product_action=None,
        hashed_url=None,
        price_bucket=None,
        number_of_category_hash=None,
        category_hash_first_level=None,
        category_hash_second_level=None,
        category_hash_third_level=None,
        description_vector=None,
        image_vector=None,
        hour=None,
        weekday=None,
        weekend=None,
        query_vector=None,
    ):
        embedding_list = [
            self.id_embeddings(input_ids),
            self.price_bucket_embeddings(price_bucket),
            self.number_of_category_hash_embeddings(number_of_category_hash),
            self.category_hash_first_level_embeddings(category_hash_first_level),
            self.category_hash_second_level_embeddings(category_hash_second_level),
            self.category_hash_third_level_embeddings(category_hash_third_level),
            description_vector,
            image_vector,
            self.elapsed_time_embeddings(elapsed_time),
            self.event_type_embeddings(event_type),
            self.product_action_embeddings(product_action),
            self.hashed_url_embeddings(hashed_url),
            self.hour_embeddings(hour),
            self.weekday_embeddings(weekday),
            self.weekend_embeddings(weekend),
            query_vector,
        ]
        embeddings = torch.cat(embedding_list, dim=-1)
        embeddings = self.linear_embed(embeddings)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        output_type: str,
        encoder_params: dict,
        preprocessor,
        dropout: float = 0.3,
        num_labels: int = 1,
    ) -> None:
        super().__init__()
        self.output_type = output_type
        self.encoder_params = encoder_params
        self.preprocessor = preprocessor
        self.encoder_params["vocab_size"] = num_labels   # number of unique items + padding id

        self.embeddings = EncoderEmbeddings(self.encoder_params, preprocessor=preprocessor)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_params["hidden_size"],
            nhead=encoder_params["nhead"],
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_params["num_layers"],
        )
        self.dropout = nn.Dropout(dropout)
        self.global_average_pooling_1d = GlobalAveragePooling1D()

        self.seq_next_item = nn.LSTM(
            input_size=encoder_params["hidden_size"],
            bidirectional=False,
            hidden_size=encoder_params["lstm_hidden_size"],
            num_layers=encoder_params["lstm_num_layers"],
            dropout=encoder_params["lstm_dropout"],
        )
        self.ffn_next_item = nn.Sequential(
            nn.Linear(encoder_params["lstm_hidden_size"], encoder_params["lstm_hidden_size"]),
            nn.LayerNorm(encoder_params["lstm_hidden_size"]),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_params["lstm_hidden_size"], num_labels),
        )
        self.seq = nn.LSTM(
            input_size=encoder_params["hidden_size"],
            bidirectional=True,
            hidden_size=encoder_params["lstm_hidden_size"],
            num_layers=encoder_params["lstm_num_layers"],
            dropout=encoder_params["lstm_dropout"],
        )
        self.sequence_embedding_subsequent_items = nn.Sequential(
            nn.Linear(encoder_params["lstm_hidden_size"] * 2, encoder_params["lstm_hidden_size"]),
            nn.LayerNorm(encoder_params["lstm_hidden_size"]),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_params["lstm_hidden_size"], encoder_params["embedding_size"]),
        )
        self.sequence_embedding_next_item = nn.Sequential(
            nn.Linear(encoder_params["lstm_hidden_size"], encoder_params["lstm_hidden_size"]),
            nn.LayerNorm(encoder_params["lstm_hidden_size"]),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_params["lstm_hidden_size"], encoder_params["embedding_size"]),
        )
        self.items_bias = nn.Parameter(torch.Tensor(num_labels,))
        self.items_bias.data.normal_(0., 0.01)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        elapsed_time=None,
        event_type=None,
        product_action=None,
        hashed_url=None,
        price_bucket=None,
        number_of_category_hash=None,
        category_hash_first_level=None,
        category_hash_second_level=None,
        category_hash_third_level=None,
        description_vector=None,
        image_vector=None,
        hour=None,
        weekday=None,
        weekend=None,
        query_vector=None,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            elapsed_time=elapsed_time,
            event_type=event_type,
            product_action=product_action,
            hashed_url=hashed_url,
            price_bucket=price_bucket,
            number_of_category_hash=number_of_category_hash,
            category_hash_first_level=category_hash_first_level,
            category_hash_second_level=category_hash_second_level,
            category_hash_third_level=category_hash_third_level,
            description_vector=description_vector,
            image_vector=image_vector,
            hour=hour,
            weekday=weekday,
            weekend=weekend,
            query_vector=query_vector,
        )
        # encoder_outputs: [batch, seq_len, d_model] => [seq_len, batch, d_model]
        embedding_output = embedding_output.permute([1, 0, 2])
        encoder_outputs = self.encoder(
            embedding_output,
        )
        encoder_outputs = self.dropout(encoder_outputs)

        if self.output_type == "subsequent_items":
            # hidden: [seq_len, batch, lstm_hidden_dim]
            hidden, _  = self.seq(encoder_outputs)
            hidden = hidden.permute([1, 0, 2])
            last_state = self.global_average_pooling_1d(hidden)
            sequence_embedding = self.sequence_embedding_subsequent_items(last_state)
            output_subsequent_items = nn.functional.linear(
                input=sequence_embedding,
                weight=self.embeddings.id_embeddings.weight,
                bias=self.items_bias,
            )
            return output_subsequent_items

        elif self.output_type == "next_item":
            # hidden: [seq_len, batch, lstm_hidden_dim]
            hidden, _  = self.seq_next_item(encoder_outputs)
            last_state = hidden[-1, :, :]
            sequence_embedding = self.sequence_embedding_next_item(last_state)
            output_next_item = nn.functional.linear(
                input=sequence_embedding,
                weight=self.embeddings.id_embeddings.weight,
                bias=self.items_bias,
            )
            return output_next_item


class GlobalAveragePooling1D(nn.Module):
    """https://discuss.pytorch.org/t/equivalent-of-keras-globalmaxpooling1d/45770/4
    """
    def __init__(self, data_format='channels_last'):
        super(GlobalAveragePooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == 'channels_last' else 2

    def forward(self, input):
        return torch.mean(input, axis=self.step_axis)
