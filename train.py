import argparse
import os.path
from os.path import split

import torch
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses.cross_entropy import CrossEntropyMasked
from models.seq2seq import EncoderRNN, DecoderRNN, Seq2Seq
from utilities.gpu_util import check_gpu_availability


def train_loop(data_loader, model, optimizer, loss_function, curr_epoch, decoder_tokenizer, args) :

    pbar = tqdm(enumerate(data_loader), total= len(data_loader), desc = f'Training \t Epoch : {curr_epoch} / {args.epochs}', unit= 'batch')
    epoch_loss = 0

    model.train()

    for idx, batch in pbar :

        # To get details of the dataset

        # print(f'Batch {idx} : ')
        # print("Encoder input shape:", batch['encoder_input'].shape)  # (batch_size, max_length_input)
        # print("Decoder input shape:", batch['decoder_input'].shape)  # (batch_size, max_length_target - 1)
        # print("Decoder output shape:", batch['decoder_output'].shape)  # (batch_size, max_length_target - 1)
        # print("Encoder mask shape:", batch['encoder_mask'].shape)   # (batch_size, max_length_input)
        # print("Decoder input mask shape:", batch['decoder_mask'].shape) # (batch_size, max_length_target - 1)
        # print("Decoder output mask shape : ", batch['decoder_output'].shape) # (batch_size, max_length_target - 1)
        #
        # print(f"-" * 30)
        # print("Encoder Details : ")
        # print(batch['encoder_input'][0])
        # print("Decoder Details : ")
        # print(batch['decoder_input'][0])
        # print("Decoder Output :")
        # print(batch['decoder_output'][0])

        # Forward pass
        encoder_input = batch['encoder_input'].to(args.device)
        decoder_input = batch['decoder_input'].to(args.device)
        decoder_output = batch['decoder_output'].to(args.device)
        decoder_output_mask = batch['decoder_mask'].to(args.device)

        if curr_epoch <= args.tf_epochs :
            # Perform Teacher Forcing
            output = model(src= encoder_input, trg= decoder_input, use_teacher_forcing=True)
        elif curr_epoch > args.tf_epochs and curr_epoch <= args.tf_epochs + 10 :
            # Perform Teach Forcing with probability of 0.5
            output = model(src=encoder_input, trg= decoder_input, use_teacher_forcing=False, use_only_predictions=False)
        else :
            # Depends completely on the predicted token to get next token
            output = model(src=encoder_input, trg= decoder_input, use_teacher_forcing=False, use_only_predictions=True)


        # Computing Loss Function
        loss = loss_function(logits = output, targets= decoder_output, mask= decoder_output_mask)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adding Details to progress bar
        batch_size = encoder_input.shape[0]

        # BLEU and ROUGE Score calculation
        bleu_scores = []
        rouge_scores = {
            'rouge-1' : [] ,
            'rouge-2' : [] ,
            'rouge-l' : []
        }

        # Decode predictions and targets
        predictions = output.argmax(-1).detach().cpu().numpy()  # Predicted tokens
        targets = decoder_output.detach().cpu().numpy()  # Target tokens

        for pred, target in zip(predictions, targets):
            # Decode tokens

            pred = pred.tolist()
            target = target.tolist()

            pred_text = [decoder_tokenizer.get_word_from_index(token) for token in pred if decoder_tokenizer.get_word_from_index(token) not in  ['<UNK>' , 'end']]
            target_text = [decoder_tokenizer.get_word_from_index(token) for token in target if decoder_tokenizer.get_word_from_index(token) not in  ['<UNK>' , 'end']]

            pred_text = " ".join(pred_text)
            target_text = " ".join(target_text)

            if not (len(pred_text) > 0 and len(target_text) > 0) :
                continue

            # BLEU Score
            bleu_scores.append(
                sentence_bleu(
                    [target_text.split()],
                    pred_text.split(),
                    smoothing_function=SmoothingFunction().method1
                )
            )

            # ROUGE Scores
            rouge = Rouge()
            scores = rouge.get_scores(pred_text, target_text)
            rouge_scores["rouge-1"].append(scores[0]["rouge-1"]['f'])
            rouge_scores["rouge-2"].append(scores[0]["rouge-2"]['f'])
            rouge_scores["rouge-l"].append(scores[0]["rouge-l"]['f'])

        # Compute average BLEU and ROUGE scores
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_rouge_1 = sum(rouge_scores["rouge-1"]) / len(rouge_scores["rouge-1"]) if rouge_scores["rouge-1"] else 0
        avg_rouge_2 = sum(rouge_scores["rouge-2"]) / len(rouge_scores["rouge-2"]) if rouge_scores["rouge-2"] else 0
        avg_rouge_l = sum(rouge_scores["rouge-l"]) / len(rouge_scores["rouge-l"]) if rouge_scores["rouge-l"] else 0

        # print(
        #     f"Avg BLEU: {avg_bleu:.4f}, Avg ROUGE-1: {avg_rouge_1:.4f}, Avg ROUGE-2: {avg_rouge_2:.4f}, Avg ROUGE-L: {avg_rouge_l:.4f}")


        pbar.set_postfix({
            'Loss': loss.item() / batch_size ,
            'Avg BLEU' : f'{avg_bleu:.4f}',
            'Avg ROUGE-1' : f'{avg_rouge_1:.4f}',
            'Avg ROUGE-2' : f'{avg_rouge_2:.4f}',
            'Avg ROUGE-L' : f'{avg_rouge_l:.4f}'
        })
        epoch_loss += loss.item()

        # Adding Results in Tensorboard
        args.writer.add_scalar('train/loss', loss.item() / batch_size, curr_epoch)
        args.writer.add_scalar('train/bleu', avg_bleu, curr_epoch)
        args.writer.add_scalar('train/rouge-1', avg_rouge_1, curr_epoch)
        args.writer.add_scalar('train/rouge-2', avg_rouge_2, curr_epoch)
        args.writer.add_scalar('train/rouge-l', avg_rouge_l, curr_epoch)


    return epoch_loss / len(data_loader)

def evaluation(data_loader, model, optimizer, loss_function, curr_epoch, encoder_tokenizer, decoder_tokenizer, args, split_type) :

    model.eval()

    if split_type not in ['val', 'test'] :
        print("Enter a valid split type e.g. val or test ")
        exit()

    pbar = tqdm(enumerate(data_loader), total= len(data_loader), desc=f'{split_type} \t Epoch : {curr_epoch} / {args.epochs}', unit='batch')
    epoch_loss = 0

    for idx, batch in pbar :

        # Forward pass
        encoder_input = batch['encoder_input'].to(args.device)
        decoder_input = batch['decoder_input'].to(args.device)
        decoder_output = batch['decoder_output'].to(args.device)
        decoder_output_mask = batch['decoder_mask'].to(args.device)

        # Depends completely on the predicted token to get next token
        output = model(src=encoder_input, trg=decoder_input, use_teacher_forcing=False, use_only_predictions=True)

        # Computing Loss Function
        loss = loss_function(logits = output, targets= decoder_output, mask= decoder_output_mask)

        # Adding Details to progress bar
        batch_size = encoder_input.shape[0]

        # BLEU and ROUGE Score calculation
        bleu_scores = []
        rouge_scores = {
            'rouge-1' : [] ,
            'rouge-2' : [] ,
            'rouge-l' : []
        }

        # Decode predictions and targets
        predictions = output.argmax(-1).detach().cpu().numpy()  # Predicted tokens
        targets = decoder_output.detach().cpu().numpy()  # Target tokens
        src_inputs = encoder_input.detach().cpu().numpy() # Source Inputs

        for pred, target, src_txt in zip(predictions, targets, src_inputs):

            # Decode tokens
            pred = pred.tolist()
            target = target.tolist()
            src_txt = src_txt.tolist()

            pred_text = [decoder_tokenizer.get_word_from_index(token) for token in pred if
                         decoder_tokenizer.get_word_from_index(token) not in ['<UNK>', 'end']]
            target_text = [decoder_tokenizer.get_word_from_index(token) for token in target if
                           decoder_tokenizer.get_word_from_index(token) not in ['<UNK>', 'end']]
            source_text = [encoder_tokenizer.get_word_from_index(token) for token in src_txt if
                           encoder_tokenizer.get_word_from_index(token) not in ['<UNK>', 'end']]

            source_text = " ".join(source_text)
            pred_text = " ".join(pred_text)
            target_text = " ".join(target_text)

            # BLEU Score
            bleu_scores.append(
                sentence_bleu(
                    [target_text.split()],
                    pred_text.split(),
                    smoothing_function=SmoothingFunction().method1
                )
            )

            # ROUGE Scores
            rouge = Rouge()
            scores = rouge.get_scores(pred_text, target_text)
            rouge_scores["rouge-1"].append(scores[0]["rouge-1"]['f'])
            rouge_scores["rouge-2"].append(scores[0]["rouge-2"]['f'])
            rouge_scores["rouge-l"].append(scores[0]["rouge-l"]['f'])

        # Compute average BLEU and ROUGE scores
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_rouge_1 = sum(rouge_scores["rouge-1"]) / len(rouge_scores["rouge-1"]) if rouge_scores["rouge-1"] else 0
        avg_rouge_2 = sum(rouge_scores["rouge-2"]) / len(rouge_scores["rouge-2"]) if rouge_scores["rouge-2"] else 0
        avg_rouge_l = sum(rouge_scores["rouge-l"]) / len(rouge_scores["rouge-l"]) if rouge_scores["rouge-l"] else 0


        pbar.set_postfix({
            'Loss': loss.item() / batch_size ,
            'Avg BLEU' : f'{avg_bleu:.4f}',
            'Avg ROUGE-1' : f'{avg_rouge_1:.4f}',
            'Avg ROUGE-2' : f'{avg_rouge_2:.4f}',
            'Avg ROUGE-L' : f'{avg_rouge_l:.4f}'
        })
        epoch_loss += loss.item()

        # Adding Results in Tensorboard
        args.writer.add_scalar(f'{split_type}/loss', loss.item() / batch_size, curr_epoch)
        args.writer.add_scalar(f'{split_type}/bleu', avg_bleu, curr_epoch)
        args.writer.add_scalar('train/rouge-1', avg_rouge_1, curr_epoch)
        args.writer.add_scalar('train/rouge-2', avg_rouge_2, curr_epoch)
        args.writer.add_scalar('train/rouge-l', avg_rouge_l, curr_epoch)

    if curr_epoch % 10 == 1 :
        print("Need to write script for some samples")

    return epoch_loss / len(data_loader)


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
        RC_UNIT = args.rc_unit  # We have 3 different types of RNN units starting with rnn

        # Initialize Encoder, Decoder, and Seq2Seq model
        encoder = EncoderRNN(INPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, RC_UNIT).to(args.device)
        decoder = DecoderRNN(OUTPUT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, RC_UNIT).to(args.device)
        seq2seq_model = Seq2Seq(encoder, decoder, args.device).to(args.device)
        args.model = seq2seq_model


        # ----------------------------------- Optimizer and Loss Function Initializer ----------------------------
        args.optimizer = optim.Adam(seq2seq_model.parameters(), lr=args.lr)

        if args.loss == 'cross_entropy' :
            args.loss_function = CrossEntropyMasked()

    # Initialize variables for saving the model and early stopping
    best_val_loss = float('inf')  # Best validation loss starts at infinity
    no_improvement_epochs = 0  # Counter for epochs without improvement
    patience = 10  # Early stopping patience (number of epochs to wait for improvement)

    # Training and Evaluation of Model
    for epoch in range(args.epochs) :

        epoch_train_loss = train_loop(data_loader= train_loader, model = args.model, optimizer= args.optimizer,
                   loss_function= args.loss_function, curr_epoch = epoch + 1, args= args,
                   decoder_tokenizer= y_tokenizer)

        epoch_val_loss = evaluation(data_loader= val_loader, model = args.model, optimizer= args.optimizer,
                                    loss_function= args.loss_function, curr_epoch= epoch + 1, args = args,
                                    decoder_tokenizer= y_tokenizer, encoder_tokenizer= x_tokenizer, split_type= 'val')

        epoch_test_loss = evaluation(data_loader= test_loader, model = args.model, optimizer= args.optimizer,
                                    loss_function= args.loss_function, curr_epoch= epoch + 1, args = args,
                                    decoder_tokenizer= y_tokenizer, encoder_tokenizer= x_tokenizer, split_type= 'test')

        print(f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")

        # Check if validation loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improvement_epochs = 0  # Reset counter if there's improvement

            # Save the model
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving model...")

            final_dict = {
                'encoder_tokenizer' : x_tokenizer,
                'decoder_tokenizer' : y_tokenizer,
                'max_encoder_length' : max_input_length,
                'max_deocder_length' : max_output_length,
                'model_name' : args.model_type,
                'model_weights' : args.model.state_dict()
            }
            ckpt_path = os.path.join(args.ckpts, args.exp_name, 'best_model.pth')
            torch.save(args.model.state_dict(), ckpt_path)

        else:
            no_improvement_epochs += 1
            print(f"No improvement for {no_improvement_epochs} epoch(s).")

        # Early stopping
        if no_improvement_epochs >= patience:
            print("Early stopping triggered. No improvement for 10 consecutive epochs.")
            break

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
    parser.add_argument('--bs', type=int, default=128, help="Batch size ")
    parser.add_argument('--loss', type = str, default= 'cross_entropy', help='Different Loss Functions to be defined')
    parser.add_argument('--epochs', type= int, default= 100, help = 'Total epochs to run')
    parser.add_argument('--tf_epochs', type= int, default= 20, help = 'Initialize to trianing model using teacher forcing ')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Defining Learning Rate for model')
    parser.add_argument('--runs', type= str, default= 'runs', help = 'Folder path where the tensorboard results are stored')
    parser.add_argument('--ckpts', type = str, default= '/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Ckpts/', help = 'Folder paths where the checkpoints are saved')
    args = parser.parse_args()

    # Assign GPU device
    gpu_ls = check_gpu_availability(required_space_gb=10, required_gpus=1)
    args.device  = f'cuda:{gpu_ls[0]}'

    # Assert Condition
    assert args.tf_epochs < args.epochs

    # Making the runs folder
    if not os.path.exists(args.runs) :
        os.makedirs(args.runs, exist_ok= True)

    # Creating exp name
    if args.model_type == 'seq2seq' :
        exp_name = f"dataset_{args.dataset_name}/model_{args.model_type}_rcu_{args.rc_unit}_loss_{args.loss}_lr_{args.lr}_emb_{args.embed_size}_hs_{args.hidden_size}_stk_{args.rc_layers}"
        args.exp_name = exp_name
    # Creating exp_name folder
    exp_path = os.path.join(args.runs, exp_name)
    os.makedirs(exp_path, exist_ok= True)

    # Creating Writer
    writer = SummaryWriter(exp_path)
    args.writer = writer

    # Argument Details
    print("-" * 20, "Arguments", "-" * 20)
    # Convert to dictionary and iterate
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    run(args)