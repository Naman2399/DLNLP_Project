import argparse

import nltk
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset.news_dataset import create_data_loader, preprocess, TextTokenizer, TextSummarizationDataset

nltk.download('stopwords')


def create_data_loader(args) :

    global stop_words

    # Load the dataset
    dataset = load_dataset('QuyenAnhDE/data-for-text-summarization')

    print(dataset)

    # Extract text and summary from the training split
    texts = dataset['train']['text']
    summaries = dataset['train']['summary']

    # Check the lengths of text and summary
    print(f"Number of text entries: {len(texts)}")
    print(f"Number of summary entries: {len(summaries)}")

    # Create a DataFrame
    df = pd.DataFrame({
        'text': texts,
        'summary': summaries
    })

    # Print dataframe columns
    print(f"Columns : {df.columns}")

    # Preprocess the dataset

    # Removing Duplicates
    df.drop_duplicates(subset=['text'], inplace=True)

    # Remove rows where either 'text' or 'summary' is None
    df = df.dropna(subset=['text', 'summary'])

    # Resetting the Index
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

    # Create datasets
    train_dataset = TextSummarizationDataset(X_train, y_train, x_tokenizer, y_tokenizer, max_length_x, max_length_y)
    val_dataset = TextSummarizationDataset(X_val, y_val, x_tokenizer, y_tokenizer, max_length_x, max_length_y)
    test_dataset = TextSummarizationDataset(X_test, y_test, x_tokenizer, y_tokenizer, max_length_x, max_length_y)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    return train_loader, val_loader, test_loader, x_tokenizer, y_tokenizer

def run(args) :

    return create_data_loader(args)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32, help="Batch size ")
    args = parser.parse_args()

    print("-" * 20, "Arguments", "-" * 20)
    # Convert to dictionary and iterate
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run(args)