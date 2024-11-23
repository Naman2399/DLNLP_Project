import argparse
import re

import contractions
import nltk
import pandas as pd
import torch
from keras.src.ops import dtype
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset, DataLoader

nltk.download('stopwords')

# Preprocess function
def preprocess(text):

    # Download necessary NLTK resources
    stop_words = stopwords.words('english')

    text = text.lower()
    text = ' '.join([contractions.fix(word) for word in text.split(" ")])
    tokens = [w for w in text.split() if w not in stop_words]
    text = " ".join(tokens)
    text = text.replace("'s", '')
    text = re.sub(r'\(.*\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9. ]', ' ', text)
    text = re.sub(r'\.', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# Tokenizer and padding
class TextTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.word_index = {}
        self.index_word = {}

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index  # Word to index mapping
        self.index_word = {v: k for k, v in self.word_index.items()}  # Index to word mapping

    def texts_to_padded_sequences(self, texts, maxlen):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=maxlen, padding='post')

    def get_vocab_size(self):
        return len(self.tokenizer.word_index) + 1

    def get_word_from_index(self, index):
        """
        Get the word corresponding to an index.
        """
        return self.index_word.get(index, "<UNK>")  # Return <UNK> if index is not found


# Dataset class
class TextSummarizationDataset(Dataset):
    def __init__(self, inputs, targets, input_tokenizer, target_tokenizer, max_length_input, max_length_target):
        assert len(inputs) == len(targets), "Inputs and targets must have the same length."
        self.inputs = inputs
        self.targets = targets
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx >= len(self.inputs):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.inputs)}.")

        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        # Tokenize and pad input (encoder sequence)
        input_sequence = self.input_tokenizer.texts_to_padded_sequences([input_text], self.max_length_input)[0]
        input_mask = [1 if token != 0 else 0 for token in input_sequence]  # Mask for non-padding tokens

        # Tokenize and pad target (decoder sequence)
        target_sequence = self.target_tokenizer.texts_to_padded_sequences([target_text], self.max_length_target)[0]
        decoder_input_sequence = target_sequence[:-1]  # Exclude the last token for decoder input
        decoder_input_mask = [1 if token != 0 else 0 for token in decoder_input_sequence]
        decoder_output_sequence = target_sequence[1:]  # Exclude the first token for decoder output
        decoder_output_mask = [1 if token != 0 else 0 for token in decoder_output_sequence]  # Mask for non-padding tokens

        return {
            'encoder_input': torch.tensor(input_sequence, dtype=torch.long),
            'encoder_mask': torch.tensor(input_mask, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input_sequence, dtype=torch.long),
            'decoder_input_mask' : torch.tensor(decoder_input_mask, dtype= torch.long),
            'decoder_output': torch.tensor(decoder_output_sequence, dtype=torch.long),
            'decoder_output_mask': torch.tensor(decoder_output_mask, dtype=torch.long)
        }


def create_data_loader(args) :

    global stop_words

    # Download necessary NLTK resources
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    # File paths
    raw_file = args.raw_csv

    # Load data
    df = pd.read_csv(raw_file, encoding='iso-8859-1')

    # Converting data frame columns
    df = df.rename(columns = {'headlines' : 'summary'})

    # Print dataframe columns
    print(f"Columns : {df.columns}")

    # Preprocess the dataset
    df.drop_duplicates(subset=['summary'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['summary'] = df['summary'].apply(preprocess)
    df['text'] = df['text'].apply(preprocess)
    df['summary'] = df['summary'].apply(lambda x: '_START_ ' + x + ' _END_')
    df.reset_index(drop=True, inplace=True)

    # Resetting indices after splitting
    X_train, X_val, y_train, y_val = train_test_split(df['text'].tolist(), df['summary'].tolist(), test_size=0.2,
                                                      random_state=20)
    X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, test_size=0.5, random_state=20)

    # Fit tokenizer
    x_tokenizer = TextTokenizer()
    x_tokenizer.fit(X_train + y_train)

    y_tokenizer = TextTokenizer()
    y_tokenizer.fit(y_train)

    max_length_x = max(len(x.split()) for x in X_train)
    max_length_y = max(len(y.split()) for y in y_train)

    # Debugging splits
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
    print(f"Max input length: {max_length_x}, Max target length: {max_length_y}")

    # Create dataset
    train_dataset = TextSummarizationDataset(X_train, y_train, x_tokenizer, y_tokenizer, max_length_x, max_length_y)
    val_dataset = TextSummarizationDataset(X_val, y_val, x_tokenizer, y_tokenizer, max_length_x, max_length_y)
    test_dataset = TextSummarizationDataset(X_test, y_test, x_tokenizer, y_tokenizer, max_length_x, max_length_y)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    # Example usage
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print("Encoder input shape:", batch['encoder_input'].shape)  # (batch_size, max_length_input)
        print("Decoder input shape:", batch['decoder_input'].shape)  # (batch_size, max_length_target - 1)
        print("Decoder output shape:", batch['decoder_output'].shape)  # (batch_size, max_length_target - 1)

        print(f"-" * 30)
        print("Encoder Details : ")
        print(batch['encoder_input'][0])
        print("Decoder Details : ")
        print(batch['decoder_input'][0])
        print("Decoder Output :")
        print(batch['decoder_output'][0])

        break

    print(f"-" * 30)
    print("Vocabulary Details")

    # Print details of the x_tokenizer (input tokenizer)
    print("Details of x_tokenizer (input):")
    print(f"Vocabulary size: {x_tokenizer.get_vocab_size()}")
    print("Sample of token-to-word mapping:")
    for word, index in list(x_tokenizer.tokenizer.word_index.items())[:10]:  # Print first 10 mappings
        print(f"Token: {index}, Word: {word}")

    # Print details of the y_tokenizer (output tokenizer)
    print("\nDetails of y_tokenizer (output):")
    print(f"Vocabulary size: {y_tokenizer.get_vocab_size()}")
    print("Sample of token-to-word mapping:")
    for word, index in list(y_tokenizer.tokenizer.word_index.items())[:10]:  # Print first 10 mappings
        print(f"Token: {index}, Word: {word}")


    return train_loader, val_loader, test_loader, x_tokenizer, y_tokenizer, max_length_x, max_length_y


def run(args) :
    return create_data_loader(args)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_csv', type = str, default = '/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary.csv', help = "summary file contents")
    parser.add_argument('--raw_csv', type = str, default = '/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary_more.csv', help = "raw csv file details")
    parser.add_argument('--bs', type = int, default = 32, help = "Batch size ")
    args = parser.parse_args()

    print("-" * 20, "Arguments", "-" * 20)
    # Convert to dictionary and iterate
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run(args)
