"""
MTL-BERT model architecture.
Transformer-encoder-based BERT for SMILES molecular representation learning.
Includes positional encoding, multi-head self-attention, and feed-forward blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Positional Encoding
# ============================================================================

class PostionalEncoding(nn.Module):
    """Sinusoidal positional encoding: injects position information into input embeddings."""

    def __init__(self, d_model, max_len, device):
        super(PostionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.encoding = self._build_encoding(max_len)

    def _build_encoding(self, max_len):
        """Build positional encoding matrix of shape [max_len, d_model]."""
        encoding = torch.zeros(max_len, self.d_model, device=self.device)
        pos = torch.arange(0, max_len, device=self.device,
                           dtype=torch.float32).unsqueeze(1)
        # Even indices → sin, odd indices → cos
        _2i = torch.arange(0, self.d_model, step=2,
                           device=self.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.d_model)))
        return encoding

    def forward(self, x):
        seq_len = x.size(1)
        # Dynamically extend encoding if sequence exceeds current max length
        if seq_len > self.encoding.size(0):
            self.encoding = self._build_encoding(seq_len)
        return self.encoding[:seq_len, :].unsqueeze(0).to(x.device)


# ============================================================================
# Attention Masks
# ============================================================================

def make_src_mask(src, src_pad_idx=0):
    """Create a padding mask for the source sequence.
    Positions with <PAD> (index 0) get attention scores set to -inf.
    """
    src_mask = (src == src_pad_idx).unsqueeze(1).unsqueeze(2).to(src.device)
    return src_mask


def make_trg_mask(trg, trg_pad_idx=0):
    """Create a combined look-ahead + padding mask for the target sequence (decoder)."""
    trg_pad_mask = (trg == trg_pad_idx).unsqueeze(1).unsqueeze(3)
    trg_len = trg.shape[1]
    # Lower-triangular matrix prevents attending to future positions
    trg_sub_mask = torch.tril(torch.ones(
        trg_len, trg_len)).type(torch.bool).to(trg.device)
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask.to(trg.device)


# ============================================================================
# Attention Mechanisms
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V."""

    def __init__(self, dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, attn_mask=None):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        # Scale by sqrt(d_k) to prevent vanishing gradients in softmax
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            K.shape[-1])

        # Fill masked positions with a large negative value → near-zero after softmax
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention: runs multiple ScaledDotProductAttention heads in parallel."""

    def __init__(self, d_model, num_heads, rate):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # Output projection
        self.fc = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=rate)
        self.dot_product_attention = ScaledDotProductAttention(
            dropout_rate=rate)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) → (B, H, S, W): split into multiple heads
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   self.num_heads, self.depth).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1,
                                   self.num_heads, self.depth).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1,
                                   self.num_heads, self.depth).transpose(1, 2)

        # context: [batch_size, n_heads, len_q, d_v]
        context, attn = self.dot_product_attention(Q, K, V, attn_mask)
        # Merge heads: (B, H, S, W) → (B, S, D)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.fc(context)
        output = self.dropout(output)
        # Pre-LN: residual connection + layer normalization
        return self.layernorm(output + residual), attn


# ============================================================================
# Feed-Forward Network
# ============================================================================

class PoswiseFeedForwardNet(nn.Module):
    """Position-wise feed-forward network: two linear layers with GELU activation."""

    def __init__(self, d_model, dff, rate):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, d_model)
        )
        self.dropout = nn.Dropout(p=rate)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """inputs: [batch_size, seq_len, d_model]"""
        residual = inputs
        output = self.fc(inputs)
        output = self.dropout(output)
        # Pre-LN: residual connection + layer normalization
        return self.layernorm(output + residual)


# ============================================================================
# Encoder
# ============================================================================

class EncoderLayer(nn.Module):
    """Single encoder layer: multi-head self-attention followed by feed-forward network."""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, num_heads, rate)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, dff, rate)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        """
        # Self-attention: Q = K = V = enc_inputs
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # Feed-forward
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    """BERT encoder: embedding layer + positional encoding + stacked EncoderLayers."""

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding=200, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.src_emb = nn.Embedding(input_vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model, 200, device=device)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate)
             for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        Returns: enc_outputs [B, src_len, d_model], list of per-layer attention weights
        """
        # Token embedding + positional encoding
        word_emb = self.src_emb(enc_inputs)
        pos_emb = self.pos_emb(enc_inputs)
        enc_outputs = word_emb + pos_emb
        # Build padding mask from input tokens (<PAD> = 0)
        enc_self_attn_mask = make_src_mask(enc_inputs)
        enc_self_attns = []
        # Forward through each encoder layer
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class EncoderForPrediction(nn.Module):
    """Encoder variant for downstream tasks: prepends prediction task tokens to the input."""

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding=200, rate=0.1, prediction_nums=0):
        super(EncoderForPrediction, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prediction_nums = prediction_nums
        self.src_emb = nn.Embedding(input_vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model, 300, device=device)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate)
             for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        src_len = prediction_nums (task tokens) + SMILES sequence length
        """
        word_emb = self.src_emb(enc_inputs)
        pos_emb = self.pos_emb(enc_inputs[:, self.prediction_nums:])

        enc_outputs = word_emb
        # Only add positional encoding to the SMILES portion (task tokens excluded)
        enc_outputs[:, self.prediction_nums:] += pos_emb

        # Build padding mask and expand to multi-head dimensions
        enc_self_attn_mask = make_src_mask(enc_inputs)
        enc_self_attn_mask = enc_self_attn_mask.repeat(
            1, self.num_heads, enc_self_attn_mask.shape[-1], 1)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# ============================================================================
# Full Models
# ============================================================================

class BertModel(nn.Module):
    """Pre-training BERT model: Encoder + MLM classification head.
    Input: masked SMILES. Output: per-position token prediction logits.
    """

    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8,
                 vocab_size=50, dropout_rate=0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               input_vocab_size=vocab_size,
                               maximum_position_encoding=200,
                               rate=dropout_rate)
        # MLM head: d_model → 2*d_model → vocab_size
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model * 2, vocab_size)

    def forward(self, x):
        x, attns = self.encoder(x)
        y = self.fc1(x)
        y = self.dropout1(y)
        y = F.gelu(y)
        y = self.fc2(y)
        return y


class PredictionModel(nn.Module):
    """Downstream prediction model: pre-trained Encoder + per-task MLP heads.
    Supports classification (BCEWithLogitsLoss) and regression (MSELoss) tasks.
    """

    def __init__(self, hidden_num, finetune, num_layers=6, d_model=256,
                 dff=512, num_heads=8, vocab_size=60, dropout_rate=0.1,
                 reg_nums=0, clf_nums=0):
        super(PredictionModel, self).__init__()

        self.reg_nums = reg_nums
        self.clf_nums = clf_nums
        self.hidden_num = hidden_num
        self.finetune = finetune

        # Encoder with prediction task tokens prepended to the input
        self.encoder = EncoderForPrediction(
            num_layers=num_layers, d_model=d_model,
            num_heads=num_heads, dff=dff, input_vocab_size=vocab_size,
            maximum_position_encoding=200, rate=dropout_rate,
            prediction_nums=self.reg_nums + self.clf_nums)

        # ---- Fine-tuning strategy ----
        if self.finetune == 'none':
            # Freeze all pre-trained parameters
            self.encoder.requires_grad_(False)

        elif self.finetune == 'partial':
            # Freeze embedding and all but the last 2 encoder layers
            for layer in self.encoder.layers[:-2]:
                for p in layer.parameters():
                    p.requires_grad = False
            for p in self.encoder.src_emb.parameters():
                p.requires_grad = False

        elif self.finetune == 'all':
            # All parameters are trainable
            self.encoder.requires_grad_(True)

        # One independent MLP head per task: d_model → hidden_num → 1
        self.fc_list = nn.ModuleList()
        for i in range(self.clf_nums + self.reg_nums):
            self.fc_list.append(nn.Sequential(
                nn.Linear(d_model, hidden_num),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_num, 1)))

    def forward(self, x):
        """Forward pass. Returns a dict: {'clf': ..., 'reg': ...}."""
        x, attns = self.encoder(x)

        # Each task head reads from its corresponding prediction token position
        ys = []
        for i in range(self.clf_nums + self.reg_nums):
            y = self.fc_list[i](x[:, i])       # encoding of the i-th task token
            ys.append(y)

        y = torch.cat(ys, dim=-1)

        # Split output into classification and regression parts
        properties = {'clf': None, 'reg': None}
        if self.clf_nums > 0:
            clf = y[:, :self.clf_nums]
            properties['clf'] = clf
        if self.reg_nums > 0:
            reg = y[:, self.clf_nums:]
            properties['reg'] = reg
        return properties
