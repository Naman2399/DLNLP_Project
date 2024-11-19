import argparse

from models.seq2seq import EncoderRNN, DecoderRNN, Seq2Seq
from utilities.gpu_util import check_gpu_availability

def train_loop() :
    return

def load_dataset(args) :
    if args.dataset_name == 'news' :
        from datasets.news_dataset import create_data_loader
        return create_data_loader(args)

    elif args.dataset_name == 'hugging_face' :
        from datasets.hugging_face_dataset import create_data_loader
        return create_data_loader(args)

    else :
        print(f'Enter a valida dataset name : {args.dataset_name}')
        exit()


def run(args)  :

    # Load Dataset
    train_loader, val_loader, test_loader, x_tokenizer, y_tokenizer, max_input_length, max_output_length = load_dataset(args)

    # Load Model
    if args.model_type == 'seq2seq' :

        # ----------------------- Define model Hyper-params ------------------------------
        INPUT_VOCAB_SIZE = x_tokenizer.get_vocab_size()
        OUTPUT_VOCAB_SIZE = y_tokenizer.get_vocab_size()
        EMBED_SIZE = args.embed_size
        HIDDEN_SIZE = args.hidden_size
        NUM_LAYERS = args.rc_layers
        MAX_INPUT_LENGTH = max_input_length  # Length of input sequences
        MAX_OUTPUT_LENGTH = max_output_length  # Length of output sequences (including <sos> and <eos>)
        BATCH_SIZE = args.bs
        RC_UNIT = args.rc_unit  # We have 3 different types of RNN units starting with rnn

        # Initialize Encoder, Decoder, and Seq2Seq model
        encoder = EncoderRNN(INPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, RC_UNIT).to(args.device)
        decoder = DecoderRNN(OUTPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, RC_UNIT).to(args.device)
        seq2seq_model = Seq2Seq(encoder, decoder, args.device).to(args.device)



if __name__ == '__main__' :

    parser = argparse.ArgumentParser()

    # ------------------- Dataset Arguments -----------------------------------------
    parser.add_argument('--dataset_name', type= str, default= 'news', help = 'Options : news, hugging_face')
    parser.add_argument('--summary_csv', type=str,
                        default='/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary.csv',
                        help="summary file contents")
    parser.add_argument('--raw_csv', type=str,
                        default='/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary_more.csv',
                        help="raw csv file details")

    # ------------------ Model Arguments ---------------------------------
    parser.add_argument('--model_type', type= str, default= 'seq2seq', help='We have different model_type e.g. seq2seq')
    parser.add_argument('--rc_unit', type= str, default= 'rnn', help = 'We have 3 options for this part e.g. rnn, gru, lstm')
    parser.add_argument('--embed_size', type=int, default = 512, help = 'Embedding Dimension for the model')
    parser.add_argument('--hidden_size', type=int, default = 512, help = 'Hidden Dimension Size')
    parser.add_argument('--rc_layers', type=str, default=1, help = 'Number of Layers in Recurrent Unit')

    # --------------------- Other Arguments ----------------------------------
    parser.add_argument('--bs', type=int, default=32, help="Batch size ")

    args = parser.parse_args()

    # Assign GPU device
    gpu_ls = check_gpu_availability(required_space_gb=10, required_gpus=1)
    args.device  = f'cuda:{gpu_ls[0]}'

    # Argument Details
    print("-" * 20, "Arguments", "-" * 20)
    # Convert to dictionary and iterate
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run(args)