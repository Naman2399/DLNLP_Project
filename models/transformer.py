import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, ff_hidden_size, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size

        # Embedding layers for input and output
        self.src_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.trg_embedding = nn.Embedding(output_vocab_size, embed_size)

        # Positional encoding
        self.src_positional_encoding = PositionalEncoding(embed_size, max_seq_len, dropout)
        self.trg_positional_encoding = PositionalEncoding(embed_size, max_seq_len, dropout)

        # Transformer module
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_hidden_size,
            dropout=dropout,
            batch_first=True
        )

        # Output layer to map the transformer's output to the vocabulary
        self.fc_out = nn.Linear(embed_size, output_vocab_size)

    def forward(self, src, trg, src_pad_idx, trg_pad_idx, trg_mask=None):
        """
        Args:
            src: (batch_size, src_len) - Input source sequences
            trg: (batch_size, trg_len) - Target sequences
            src_pad_idx: Index of the padding token in the source vocabulary
            trg_pad_idx: Index of the padding token in the target vocabulary
            trg_mask: (tgt_len, tgt_len) - Optional causal mask for target sequence

        Returns:
            output: (batch_size, trg_len, output_vocab_size) - Predicted logits
        """
        # Padding masks for source and target
        src_key_padding_mask = (src == src_pad_idx)  # Shape: (batch_size, src_len)
        trg_key_padding_mask = (trg == trg_pad_idx)  # Shape: (batch_size, trg_len)

        # Embed the input and target sequences
        src_emb = self.src_embedding(src) * torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))
        trg_emb = self.trg_embedding(trg) * torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))

        # Add positional encodings
        src_emb = self.src_positional_encoding(src_emb)
        trg_emb = self.trg_positional_encoding(trg_emb)

        # Pass through the transformer
        transformer_output = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            tgt_mask=trg_mask
        )

        # Final output projection
        output = self.fc_out(transformer_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
        pe = torch.zeros(max_seq_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_size)

        Returns:
            x: Input with positional encodings added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def generate_square_subsequent_mask(size):
    """
    Generates a causal mask to prevent attention to future positions.
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))


# Testing the Model
def main():
    # Define parameters
    INPUT_VOCAB_SIZE = 1000  # Source vocabulary size
    OUTPUT_VOCAB_SIZE = 1000  # Target vocabulary size
    EMBED_SIZE = 512  # Embedding size
    NUM_HEADS = 8  # Number of attention heads
    NUM_ENCODER_LAYERS = 6  # Number of encoder layers
    NUM_DECODER_LAYERS = 6  # Number of decoder layers
    FF_HIDDEN_SIZE = 2048  # Feedforward hidden size
    MAX_SEQ_LEN = 50  # Maximum sequence length
    DROPOUT = 0.1  # Dropout probability
    BATCH_SIZE = 16  # Batch size
    SRC_PAD_IDX = 0  # Padding index for source
    TRG_PAD_IDX = 0  # Padding index for target
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TransformerModel(
        input_vocab_size=INPUT_VOCAB_SIZE,
        output_vocab_size=OUTPUT_VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        ff_hidden_size=FF_HIDDEN_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    # Dummy input and target sequences
    src = torch.randint(1, INPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN)).to(DEVICE)
    trg = torch.randint(1, OUTPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN)).to(DEVICE)

    # Generate a causal mask for the target
    trg_mask = generate_square_subsequent_mask(MAX_SEQ_LEN).to(DEVICE)

    # Forward pass
    output = model(src, trg, SRC_PAD_IDX, TRG_PAD_IDX, trg_mask=trg_mask)
    print("Output shape:", output.shape)  # Expected: (BATCH_SIZE, MAX_SEQ_LEN, OUTPUT_VOCAB_SIZE)


if __name__ == "__main__":
    main()
