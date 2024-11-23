import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, num_layers, rc_unit, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.rc_unit = rc_unit

        if self.rc_unit == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rc_unit == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rc_unit == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            print("Enter a valid RC unit name e.g. rnn, gru, lstm")
            exit()

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, seq_len, hidden_size)
        return outputs, hidden


# Define Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # Combine encoder and decoder hidden states
        self.v = nn.Parameter(torch.rand(hidden_size))      # Scoring vector

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: (batch_size, hidden_size) - Decoder hidden state at current step
            encoder_outputs: (batch_size, seq_len, hidden_size) - All encoder hidden states
        Returns:
            attn_weights: (batch_size, seq_len) - Attention weights for each input token
        """
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # Repeat decoder hidden state for each encoder step
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_size)
        energy = energy.matmul(self.v)  # (batch_size, seq_len)
        return F.softmax(energy, dim=1)  # Normalize attention weights


# Define the Decoder with Attention
class DecoderRNN(nn.Module):
    def __init__(self, output_vocab_size, embed_size, hidden_size, num_layers, rc_unit, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_vocab_size, embed_size)
        self.rc_unit = rc_unit

        if self.rc_unit == 'rnn':
            self.rnn = nn.RNN(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rc_unit == 'gru':
            self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rc_unit == 'lstm':
            self.rnn = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            print("Enter a valid RC unit name e.g. rnn, gru, lstm")
            exit()

        self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        """
        Args:
            x: (batch_size, 1) - Current input token
            hidden: (batch_size, hidden_size) - Decoder hidden state
            encoder_outputs: (batch_size, seq_len, hidden_size) - Encoder hidden states
        Returns:
            predictions: (batch_size, output_vocab_size) - Predicted logits for the current step
            hidden: Updated decoder hidden state
        """
        if self.rc_unit in ['rnn', 'gru'] :
            embedded = self.embedding(x)  # (batch_size, 1, embed_size)
            attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)  # (batch_size, seq_len)
            context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # (batch_size, 1, hidden_size)

        elif self.rc_unit in ['lstm'] :
            embedded = self.embedding(x)  # (batch_size, 1, embed_size)
            h1, c1 = hidden
            attn_weights = self.attention(h1.squeeze(0), encoder_outputs)  # (batch_size, seq_len)
            context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # (batch_size, 1, hidden_size)

        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_size + hidden_size)
        outputs, hidden = self.rnn(rnn_input, hidden)  # outputs: (batch_size, 1, hidden_size)
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
        encoder_outputs, hidden = self.encoder(src)

        # Initialize first decoder input as <sos> token
        input = trg[:, 0] # (batch_size)

        for t in range(1, trg_len):

            # Pass the input and hidden state to the decoder
            output, hidden = self.decoder(input.unsqueeze(1), hidden, encoder_outputs)
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
    RC_UNIT = 'gru' # We have 3 different types of RNN units starting with rnn

    # Create device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Encoder, Decoder, and Seq2Seq model
    encoder = EncoderRNN(INPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, RC_UNIT).to(DEVICE)
    decoder = DecoderRNN(OUTPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, RC_UNIT).to(DEVICE)
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

    # Variables
    # 1. Recurrent Unit Type,   2. Num of Layers,  3. batch_first = True
    # N - Batch Size, L = sequence Length, H_in = input_size , H_out = hidden size
    # RUT - RNN, GRU, LSTM