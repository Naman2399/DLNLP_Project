import torch
import torch.nn as nn

# Define the Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, num_layers, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, x_mask):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, seq_len, hidden_size), hidden: (num_layers, batch_size, hidden_size)
        return outputs, hidden


# Define the Decoder
class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size, num_layers, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, x, hidden):
        # x: (batch_size, 1), hidden: (num_layers, batch_size, hidden_size)
        embedded = self.embedding(x)  # (batch_size, 1, embed_size)
        outputs, hidden = self.rnn(embedded, hidden)  # outputs: (batch_size, 1, hidden_size)
        predictions = self.fc(outputs.squeeze(1))  # (batch_size, output_vocab_size)
        return predictions, hidden


# Define the Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5, use_teacher_forcing=True, use_only_predictions=False):
        """
        Args:
            src (torch.Tensor): Input source sequences of shape (batch_size, src_len)
            trg (torch.Tensor): Target sequences of shape (batch_size, trg_len)
            teacher_forcing_ratio (float): Probability of using teacher forcing.
            use_teacher_forcing (bool): If True, always use teacher forcing.
            use_only_predictions (bool): If True, rely only on predicted outputs.
        """
        # Get batch size and target sequence length
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encoder pass
        encoder_outputs, hidden = self.encoder(src, None)

        # First input to the decoder is the <sos> token
        input = trg[:, 0]  # (batch_size)

        for t in range(1, trg_len):
            # Pass the input and hidden state to the decoder
            output, hidden = self.decoder(input.unsqueeze(1), hidden)  # input.unsqueeze(1): (batch_size, 1)
            outputs[:, t, :] = output

            # Compute the predicted token
            top1 = output.argmax(1)  # (batch_size)

            if use_only_predictions:
                # Use only the predicted token as input for the next step
                input = top1
            elif use_teacher_forcing:
                # Always use the ground truth as the next input
                input = trg[:, t]
            else:
                # Use teacher forcing with a probability or the predicted token
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input = trg[:, t] if teacher_force else top1

        return outputs


def main():
    # Define dummy parameters
    INPUT_VOCAB_SIZE = 10  # Small dummy vocabulary size for input
    OUTPUT_VOCAB_SIZE = 10  # Small dummy vocabulary size for output
    EMBED_SIZE = 16  # Embedding size
    HIDDEN_SIZE = 32  # Hidden size of GRU
    NUM_LAYERS = 1  # Single GRU layer
    MAX_INPUT_LENGTH = 5  # Length of input sequences
    MAX_OUTPUT_LENGTH = 6  # Length of output sequences (including <sos> and <eos>)
    BATCH_SIZE = 2  # Number of samples in a batch

    # Create device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Encoder, Decoder, and Seq2Seq model
    encoder = EncoderRNN(INPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    decoder = DecoderRNN(OUTPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    seq2seq_model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # Dummy input and target sequences
    # Batch size = 2, Sequence length = MAX_INPUT_LENGTH / MAX_OUTPUT_LENGTH
    dummy_input = torch.randint(1, INPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_INPUT_LENGTH)).to(DEVICE)  # Random integers as tokens
    dummy_output = torch.randint(1, OUTPUT_VOCAB_SIZE, (BATCH_SIZE, MAX_OUTPUT_LENGTH)).to(DEVICE)  # Random integers as tokens

    print("Dummy Input:")
    print(dummy_input)
    print("\nDummy Output (Target):")
    print(dummy_output)

    # Forward pass through the model
    output = seq2seq_model(dummy_input, dummy_output)

    # Print model output shape
    print("\nModel Output Shape:", output.shape)  # Should be (BATCH_SIZE, MAX_OUTPUT_LENGTH, OUTPUT_VOCAB_SIZE)

    # Print the first sequence's predicted probabilities
    print("\nFirst sequence predicted probabilities:")
    print(output[0])  # Probabilities over vocabulary for each time step

if __name__ == "__main__":
    main()
