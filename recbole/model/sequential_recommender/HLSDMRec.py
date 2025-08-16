import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention
from recbole.model.loss import BPRLoss
from mamba_ssm.models.mixer_seq_simple import Mamba


class GARec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GARec, self).__init__(config, dataset)
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']

        # 对比学习相关参数
        self.contrastive_temp = config.get('contrastive_temp', 0.1)  # InfoNCE温度参数
        self.contrastive_weight = config.get('contrastive_weight', 0.1)  # 对比学习损失权重

        self.q = 2  # 短期序列长度

        # define layers
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )
        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)

        # Mamba for long-term encoding
        self.mamba_layer = Mamba(
            d_model=self.hidden_size,
            n_layers=2,
            dropout=self.hidden_dropout_prob
        )

        # LSTM for short-term encoding
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=False
        )

        # Conv1D for short-term preprocessing
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            padding=1
        )


        self.long_term_item_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.long_term_feature_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.short_term_item_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.short_term_feature_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Gating mechanisms for long-term and short-term fusion
        self.item_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
            nn.Softmax(dim=-1)
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.final_layer = nn.Linear(self.hidden_size * 4, self.hidden_size)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def fft_ifft_mamba(self, x):
        x_freq = torch.fft.rfft(x, dim=1)
        freq_size = x_freq.size(1)
        hidden_size = x.size(2)

        if not hasattr(self, 'freq_filter'):
            self.freq_filter = nn.Parameter(
                torch.randn(1, freq_size, hidden_size, dtype=torch.cfloat, device=x.device)
            )

        x_filtered = x_freq * self.freq_filter
        x_time = torch.fft.irfft(x_filtered, n=x.size(1), dim=1)
        x_out = self.mamba_layer(x_time)
        return x_out

    def conv1d_lstm_encoder(self, short_term_seq):
        x = short_term_seq.transpose(1, 2)
        x = self.conv1d(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        return output

    def infonce_loss(self, item_emb, feature_emb, temperature=0.1):

        batch_size = item_emb.size(0)

        # L2标准化
        item_emb = F.normalize(item_emb, p=2, dim=1)
        feature_emb = F.normalize(feature_emb, p=2, dim=1)

        similarity = torch.matmul(item_emb, feature_emb.transpose(0, 1)) / temperature

        labels = torch.arange(batch_size, device=item_emb.device)


        loss_i2f = F.cross_entropy(similarity, labels)

        loss_f2i = F.cross_entropy(similarity.transpose(0, 1), labels)

        # 总的对比学习损失
        contrastive_loss = (loss_i2f + loss_f2i) / 2

        return contrastive_loss

    def forward(self, item_seq, item_seq_len, user_id):
        batch_size, seq_len = item_seq.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # User embeddings
        user_emb = self.user_embedding(user_id)

        # Item embeddings
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Feature embeddings
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding['item']
        dense_embedding = dense_embedding['item']
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        feature_table = torch.cat(feature_table, dim=-2)
        feature_emb, _ = self.feature_att_layer(feature_table)
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        feature_emb = self.dropout(feature_emb)

        # Split sequences into long-term and short-term
        split_point = max(1, seq_len - self.q)

        # Long-term: [0, split_point)
        long_term_item = input_emb[:, :split_point, :]
        long_term_feature = feature_emb[:, :split_point, :]

        # Short-term: [split_point, seq_len)
        short_term_item = input_emb[:, split_point:, :]
        short_term_feature = feature_emb[:, split_point:, :]

        # Long-term encoding with FFT + Mamba
        if long_term_item.size(1) > 0:
            long_term_item_encoded = self.fft_ifft_mamba(long_term_item)
            long_term_feature_encoded = self.fft_ifft_mamba(long_term_feature)
            long_term_item_final = self.gather_indexes(long_term_item_encoded,
                                                       torch.full((batch_size,), split_point - 1,
                                                                  dtype=torch.long, device=item_seq.device))
            long_term_feature_final = self.gather_indexes(long_term_feature_encoded,
                                                          torch.full((batch_size,), split_point - 1,
                                                                     dtype=torch.long, device=item_seq.device))
        else:
            long_term_item_final = torch.zeros(batch_size, self.hidden_size, device=item_seq.device)
            long_term_feature_final = torch.zeros(batch_size, self.hidden_size, device=item_seq.device)

        # Short-term encoding with Conv1D + LSTM
        if short_term_item.size(1) > 0:
            short_term_item_encoded = self.conv1d_lstm_encoder(short_term_item)
            short_term_feature_encoded = self.conv1d_lstm_encoder(short_term_feature)
            short_term_item_final = self.gather_indexes(short_term_item_encoded,
                                                        torch.full((batch_size,), short_term_item.size(1) - 1,
                                                                   dtype=torch.long, device=item_seq.device))
            short_term_feature_final = self.gather_indexes(short_term_feature_encoded,
                                                           torch.full((batch_size,), short_term_feature.size(1) - 1,
                                                                      dtype=torch.long, device=item_seq.device))
        else:
            short_term_item_final = torch.zeros(batch_size, self.hidden_size, device=item_seq.device)
            short_term_feature_final = torch.zeros(batch_size, self.hidden_size, device=item_seq.device)


        long_item_proj = self.long_term_item_proj(long_term_item_final)
        long_feature_proj = self.long_term_feature_proj(long_term_feature_final)
        long_contrastive_loss = self.infonce_loss(long_item_proj, long_feature_proj, self.contrastive_temp)

        short_item_proj = self.short_term_item_proj(short_term_item_final)
        short_feature_proj = self.short_term_feature_proj(short_term_feature_final)
        short_contrastive_loss = self.infonce_loss(short_item_proj, short_feature_proj, self.contrastive_temp)

        self.long_contrastive_loss = long_contrastive_loss
        self.short_contrastive_loss = short_contrastive_loss

        # Gating mechanisms for dynamic weighting
        # Item gating: combine user, long-term and short-term item representations
        item_gate_input = torch.cat([user_emb, long_term_item_final, short_term_item_final], dim=-1)
        item_weights = self.item_gate(item_gate_input)  # [B, 2]

        # Feature gating: combine user, long-term and short-term feature representations
        feature_gate_input = torch.cat([user_emb, long_term_feature_final, short_term_feature_final], dim=-1)
        feature_weights = self.feature_gate(feature_gate_input)  # [B, 2]

        # Apply gating weights
        weighted_long_item = long_term_item_final * item_weights[:, 0:1]
        weighted_short_item = short_term_item_final * item_weights[:, 1:2]
        weighted_long_feature = long_term_feature_final * feature_weights[:, 0:1]
        weighted_short_feature = short_term_feature_final * feature_weights[:, 1:2]

        # Final user interest representation by concatenating all four embeddings
        output = torch.cat([weighted_long_item, weighted_short_item,
                            weighted_long_feature, weighted_short_feature], dim=-1)
        output = self.final_layer(output)
        output = self.LayerNorm(output)
        output = self.dropout(output)

        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, item_seq_len, user_id)

        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            main_loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.loss_fct(logits, pos_items)

        # 添加对比学习损失
        total_loss = main_loss + self.contrastive_weight * (self.long_contrastive_loss + self.short_contrastive_loss)

        return total_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, user_id)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_id = interaction[self.USER_ID]
        seq_output = self.forward(item_seq, item_seq_len, user_id)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores