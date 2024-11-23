import argparse
import os.path
from os.path import split

import pandas as pd
import torch
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.text import bleu_score
from tqdm import tqdm

from losses.cross_entropy import CrossEntropyMasked
from models.seq2seq import EncoderRNN, DecoderRNN, Seq2Seq
from models.transformer import TransformerModel, generate_square_subsequent_mask
from utilities.gpu_util import check_gpu_availability
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import json
from transformers import AutoTokenizer, PegasusForConditionalGeneration



def train_loop_seq2seq(data_loader, model, optimizer, loss_function, curr_epoch, decoder_tokenizer, args) :

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
        decoder_output_mask = batch['decoder_output_mask'].to(args.device)

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

def evaluation_loop_seq2seq(data_loader, model, optimizer, loss_function, curr_epoch, encoder_tokenizer, decoder_tokenizer, args, split_type) :

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
        decoder_output_mask = batch['decoder_output_mask'].to(args.device)

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

        for pred, target in zip(predictions, targets):

            # Decode tokens
            pred = pred.tolist()
            target = target.tolist()

            pred_text = [decoder_tokenizer.get_word_from_index(token) for token in pred if
                         decoder_tokenizer.get_word_from_index(token) not in ['<UNK>', 'end']]
            target_text = [decoder_tokenizer.get_word_from_index(token) for token in target if
                           decoder_tokenizer.get_word_from_index(token) not in ['<UNK>', 'end']]

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

        file_path = os.path.join(args.runs, args.exp_name, 'sample.txt')

        # Write the string to the file
        top_name = '-' * 20 + f'{split_type} : Epoch {curr_epoch}' + '-' * 20
        with open(file_path, "a") as file:
            file.write(top_name + "\n")

        for batch in data_loader :
            # Forward pass
            encoder_input = batch['encoder_input'].to(args.device)
            decoder_input = batch['decoder_input'].to(args.device)
            decoder_output = batch['decoder_output'].to(args.device)
            decoder_output_mask = batch['decoder_output_mask'].to(args.device)

            # Depends completely on the predicted token to get next token
            output = model(src=encoder_input, trg=decoder_input, use_teacher_forcing=False, use_only_predictions=True)

            # Decode predictions and targets
            predictions = output.argmax(-1).detach().cpu().numpy()  # Predicted tokens
            targets = decoder_output.detach().cpu().numpy()  # Target tokens
            src_inputs = encoder_input.detach().cpu().numpy()  # Source Inputs

            cnt = 0
            for pred, target, src_txt in zip(predictions, targets, src_inputs):

                cnt += 1

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

                data = f'Source : {source_text} \nTarget : {target_text} \nPrediction : {pred_text}'
                with open(file_path, "a") as file:
                    file.write(data + "\n")

                if cnt == 5 :
                    break

            break

    return epoch_loss / len(data_loader)

def train_loop_transformer(data_loader, model, curr_epoch, max_seq_len) :

    pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                desc=f'Training \t Epoch : {curr_epoch} / {args.epochs}', unit='batch')

    epoch_loss = 0

    model.train()

    for idx, batch in pbar:

        print(f'Batch {idx} : ')
        print("Encoder input shape:", batch['encoder_input'].shape)  # (batch_size, max_length_input)
        print("Decoder input shape:", batch['decoder_input'].shape)  # (batch_size, max_length_target - 1)
        print("Decoder output shape:", batch['decoder_output'].shape)  # (batch_size, max_length_target - 1)
        print("Encoder mask shape:", batch['encoder_mask'].shape)   # (batch_size, max_length_input)
        print("Decoder input mask shape:", batch['decoder_input_mask'].shape) # (batch_size, max_length_target - 1)
        print("Decoder output mask shape : ", batch['decoder_output_mask'].shape) # (batch_size, max_length_target - 1)

        print(f"-" * 30)
        print("Encoder Details : ")
        print(batch['encoder_input'][0])
        print("Decoder Details : ")
        print(batch['decoder_input'][0])
        print("Decoder Output :")
        print(batch['decoder_output'][0])

        # Forward Pass

        encoder_input = batch['encoder_input'].to(args.device)
        decoder_input = batch['decoder_input'].to(args.device)
        decoder_output = batch['decoder_output'].to(args.device)
        decoder_output_mask = batch['decoder_output_mask'].to(args.device)

        # Generate a causal mask for the target
        trg_mask = generate_square_subsequent_mask(max_seq_len).to(args.device)
        trg_mask = trg_mask.unsqueeze(0).repeat(encoder_input.shape[0], 1, 1)

        SRC_PAD_IDX = 0  # Padding index for source
        TRG_PAD_IDX = 0  # Padding index for target

        # Forward pass
        output = model(encoder_input, decoder_input, SRC_PAD_IDX, TRG_PAD_IDX, trg_mask=trg_mask)
        print("Output shape:", output.shape)  # Expected: (BATCH_SIZE, MAX_SEQ_LEN, OUTPUT_VOCAB_SIZE)

        print("OK")

def load_dataset(args) :
    if args.dataset_name == 'news' :
        from dataset.news_dataset import create_data_loader
        return create_data_loader(args)

    elif args.dataset_name == 'hugging_face' :
        from dataset.hugging_face_dataset import create_data_loader
        return create_data_loader(args)

    else :
        print(f'Enter a valida dataset name : {args.dataset_name}')
        exit()

def gigaword_dataset(args) :

    # Load Model
    if args.model_type == 't5_small':
        tokenizer = AutoTokenizer.from_pretrained("RenZHU/t5-small-finetuned-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("RenZHU/t5-small-finetuned-xsum")

    if args.model_type == 't5' :
        tokenizer = AutoTokenizer.from_pretrained("sysresearch101/t5-large-finetuned-xsum-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("sysresearch101/t5-large-finetuned-xsum-cnn")

    if args.model_type == 'roberta' :
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/roberta_shared_bbc_xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/roberta_shared_bbc_xsum")

    if args.model_type == 'bart_large' :
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")

    if args.model_type == 'bart_large_sum' :
        tokenizer = AutoTokenizer.from_pretrained("lidiya/bart-large-xsum-samsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("lidiya/bart-large-xsum-samsum")

    if args.model_type == 'distill_bart' :
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-1")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-1")
        
    if args.model_type == 'pegasus' :
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    from dataset.gigaword import load_dataset
    df = load_dataset() # Here dataset contains two columns "document" and "summary".
    bleu = []
    rouge1 = []
    rouge2 = []
    rougel = []
    inference_time = []
    cnt = 0

    # Iterate over the rows line by line
    for index, row in df.iterrows():
        cnt += 1
        document = row['document']  # Access the 'document' column
        summary = row['summary']  # Access the 'summary' column

        print(f"Line {index + 1}:")
        print(f"Document: {document}")
        print(f"Summary: {summary}")
        print("-" * 30)

        inputs = tokenizer(document, max_length=1024, return_tensors="pt")
        # Generate Summary
        start = time.time()
        summary_ids = model.generate(inputs["input_ids"])
        end = time.time()

        outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Get Details of the following file
        # Inference Time, BELU, ROUGE1, ROUGE2, ROUGE-L

        pred_text = outputs
        target_text = summary

        # BLEU Score
        bleu.append(
            sentence_bleu(
                [target_text.split()],
                pred_text.split(),
                smoothing_function=SmoothingFunction().method1
            )
        )

        # ROUGE Scores
        rouge = Rouge()
        scores = rouge.get_scores(pred_text, target_text)
        rouge1.append(scores[0]["rouge-1"]['f'])
        rouge2.append(scores[0]["rouge-2"]['f'])
        rougel.append(scores[0]["rouge-l"]['f'])
        inference_time.append(end - start)

    data_dict = {
        'bleu' : sum(bleu) / cnt,
        'rouge1' : sum(rouge1) / cnt,
        'rouge2' : sum(rouge2) / cnt,
        'rougel' : sum(rougel) / cnt,
        'inference_time' : sum(inference_time) / cnt,
        'total params' : total_params
    }

    # Write to a JSON file
    file_path = os.path.join(args.result, args.exp_name, "output_metrics.json")
    os.makedirs(os.path.join(args.result, args.exp_name), exist_ok= True)
    with open(file_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)

    print("Data successfully written to output_metrics.json")

    return

def xsum_dataset(args) :
    # Load Model
    if args.model_type == 't5_small':
        tokenizer = AutoTokenizer.from_pretrained("RenZHU/t5-small-finetuned-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("RenZHU/t5-small-finetuned-xsum")

    if args.model_type == 't5':
        tokenizer = AutoTokenizer.from_pretrained("sysresearch101/t5-large-finetuned-xsum-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("sysresearch101/t5-large-finetuned-xsum-cnn")

    if args.model_type == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/roberta_shared_bbc_xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/roberta_shared_bbc_xsum")

    if args.model_type == 'bart_large':
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")

    if args.model_type == 'bart_large_sum':
        tokenizer = AutoTokenizer.from_pretrained("lidiya/bart-large-xsum-samsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("lidiya/bart-large-xsum-samsum")

    if args.model_type == 'distill_bart' :
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-1")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-1")

    if args.model_type == 'pegasus':
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    df = pd.read_csv('dataset/xsum/xsum.csv')
    bleu = []
    rouge1 = []
    rouge2 = []
    rougel = []
    inference_time = []
    cnt = 0

    print(df.columns)
    print(df.head())

    for index, row in df.iterrows():
        cnt += 1
        document = row['input']  # Access the 'document' column
        summary = row['target']  # Access the 'summary' column

        print(f"Line {index + 1}:")
        print(f"Document: {document}")
        print(f"Summary: {summary}")
        print("-" * 30)

        inputs = tokenizer(document, max_length=1024, return_tensors="pt")
        # Generate Summary
        start = time.time()
        summary_ids = model.generate(inputs["input_ids"])
        end = time.time()

        outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Get Details of the following file
        # Inference Time, BELU, ROUGE1, ROUGE2, ROUGE-L

        pred_text = outputs
        target_text = summary

        # BLEU Score
        bleu.append(
            sentence_bleu(
                [target_text.split()],
                pred_text.split(),
                smoothing_function=SmoothingFunction().method1
            )
        )

        # ROUGE Scores
        rouge = Rouge()
        scores = rouge.get_scores(pred_text, target_text)
        rouge1.append(scores[0]["rouge-1"]['f'])
        rouge2.append(scores[0]["rouge-2"]['f'])
        rougel.append(scores[0]["rouge-l"]['f'])
        inference_time.append(end - start)

        break

    data_dict = {
        'bleu': sum(bleu) / cnt,
        'rouge1': sum(rouge1) / cnt,
        'rouge2': sum(rouge2) / cnt,
        'rougel': sum(rougel) / cnt,
        'inference_time': sum(inference_time) / cnt,
        'total params': total_params
    }

    # Write to a JSON file
    file_path = os.path.join(args.result, args.exp_name, "output_metrics.json")
    os.makedirs(os.path.join(args.result, args.exp_name), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)

    print("Data successfully written to output_metrics.json")

    return


def run(args)  :

    if args.dataset_name in ['gigaword'] :
        gigaword_dataset(args)
        return

    if args.dataset_name in ['xsum'] :
        xsum_dataset(args)
        return

    # Load Dataset
    train_loader, val_loader, test_loader, x_tokenizer, y_tokenizer, max_input_length, max_output_length = load_dataset(args)

    # Load Model
    if args.model_type in ['seq2seq', 'seq2seq_attn'] :

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
        for epoch in range(args.epochs):

            epoch_train_loss = train_loop_seq2seq(data_loader=train_loader, model=args.model, optimizer=args.optimizer,
                                          loss_function=args.loss_function, curr_epoch=epoch + 1, args=args,
                                          decoder_tokenizer=y_tokenizer)

            epoch_val_loss = evaluation_loop_seq2seq(data_loader=val_loader, model=args.model, optimizer=args.optimizer,
                                        loss_function=args.loss_function, curr_epoch=epoch + 1, args=args,
                                        decoder_tokenizer=y_tokenizer, encoder_tokenizer=x_tokenizer, split_type='val')

            epoch_test_loss = evaluation_loop_seq2seq(data_loader=test_loader, model=args.model, optimizer=args.optimizer,
                                         loss_function=args.loss_function, curr_epoch=epoch + 1, args=args,
                                         decoder_tokenizer=y_tokenizer, encoder_tokenizer=x_tokenizer,
                                         split_type='test')

            print(
                f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")

            # Check if validation loss improved
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                no_improvement_epochs = 0  # Reset counter if there's improvement

                # Save the model
                print(f"Validation loss improved to {best_val_loss:.4f}. Saving model...")

                final_dict = {
                    'encoder_tokenizer': x_tokenizer,
                    'decoder_tokenizer': y_tokenizer,
                    'max_encoder_length': max_input_length,
                    'max_deocder_length': max_output_length,
                    'model_name': args.model_type,
                    'model_weights': args.model.state_dict()
                }
                ckpt_path = os.path.join(args.ckpts, args.exp_name, 'best_model.pth')
                os.makedirs(os.path.join(args.ckpts, args.exp_name), exist_ok=True)
                torch.save(args.model.state_dict(), ckpt_path)

            else:
                no_improvement_epochs += 1
                print(f"No improvement for {no_improvement_epochs} epoch(s).")

            # Early stopping
            if no_improvement_epochs >= patience:
                print("Early stopping triggered. No improvement for 10 consecutive epochs.")
                break

    elif args.model_type == 'transformer' :

        # ----------------------- Define model Hyper-params ------------------------------
        # Define dummy parameters
        INPUT_VOCAB_SIZE = x_tokenizer.get_vocab_size()  # Vocabulary size for input
        OUTPUT_VOCAB_SIZE = y_tokenizer.get_vocab_size()  # Vocabulary size for output
        EMBED_SIZE = args.embed_size  # Embedding size (d_model in transformer)
        NUM_HEADS = args.heads  # Number of attention heads
        NUM_ENCODER_LAYERS = args.num_encode_layers # Number of encoder layers
        NUM_DECODER_LAYERS = args.num_decode_layers # Number of decoder layers
        FF_HIDDEN_SIZE = args.hidden_size  # Feedforward hidden size
        MAX_SEQ_LEN = max_output_length  # Maximum sequence length
        DROPOUT = args.dropout  # Dropout rate
        BATCH_SIZE = args.bs  # Batch size
        DEVICE = args.device

        # Initialize the Transformer model
        transformer_model = TransformerModel(
            input_vocab_size= INPUT_VOCAB_SIZE,
            output_vocab_size= OUTPUT_VOCAB_SIZE,
            embed_size= EMBED_SIZE,
            num_heads= NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ff_hidden_size=FF_HIDDEN_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            dropout=DROPOUT
        )
        args.model= transformer_model

        # ----------------------------------- Optimizer and Loss Function Initializer ----------------------------
        args.optimizer = optim.Adam(transformer_model.parameters(), lr=args.lr)

        if args.loss == 'cross_entropy':
            args.loss_function = CrossEntropyMasked()

        # Initialize variables for saving the model and early stopping
        best_val_loss = float('inf')  # Best validation loss starts at infinity
        no_improvement_epochs = 0  # Counter for epochs without improvement
        patience = 10  # Early stopping patience (number of epochs to wait for improvement)

        # Training and Evaluation of Model
        for epoch in range(args.epochs):

            epoch_train_loss = train_loop_transformer(data_loader=train_loader, model=args.model, curr_epoch = epoch,
                                                      max_seq_len = MAX_SEQ_LEN)


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()

    # ------------------- Dataset Arguments -----------------------------------------
    parser.add_argument('--dataset_name', type= str, default= 'xsum', help = 'Options : news, hugging_face, gigaword, xsum')
    parser.add_argument('--summary_csv', type=str,
                        default='/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary.csv',
                        help="summary file contents")
    parser.add_argument('--raw_csv', type=str,
                        default='/mnt/hdd/karmpatel/naman/demo/DLNLP_Project_Data/news/news_summary_more.csv',
                        help="raw csv file details")

    # ------------------ Model Arguments ---------------------------------
    parser.add_argument('--model_type', type= str, default= 'pegasus',
                        help='We have different model_type e.g. seq2seq, seq2seq_attn, transformer, '
                             't5_small, t5, roberta, bart_large, bart_large_sum, distill_bart, pegasus')
    parser.add_argument('--rc_unit', type= str, default= 'lstm', help = 'We have 3 options for this part e.g. rnn, gru, lstm')
    parser.add_argument('--embed_size', type=int, default = 512, help = 'Embedding Dimension for the model')
    parser.add_argument('--hidden_size', type=int, default = 2048, help = 'Hidden Dimension Size')
    parser.add_argument('--rc_layers', type=str, default=5, help = 'Number of Layers in Recurrent Unit')
    parser.add_argument('--heads', type=int, default=4, help ='Number of heads in transformer')
    parser.add_argument('--num_encode_layers', type=int, default=4, help = 'Number of encoder layers in transformer')
    parser.add_argument('--num_decode_layers', type=int, default=4, help = 'Number of decoder layers in transformer')
    parser.add_argument('--dropout', type =int, default = 0.1, help = 'Dropout unit')

    # --------------------- Other Arguments ----------------------------------
    parser.add_argument('--bs', type=int, default=128, help="Batch size ")
    parser.add_argument('--loss', type = str, default= 'cross_entropy', help='Different Loss Functions to be defined')
    parser.add_argument('--epochs', type= int, default= 100, help = 'Total epochs to run')
    parser.add_argument('--tf_epochs', type= int, default= 20, help = 'Initialize to trianing model using teacher forcing ')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Defining Learning Rate for model')
    parser.add_argument('--runs', type= str, default= 'runs', help = 'Folder path where the tensorboard results are stored')
    parser.add_argument('--result', type = str, default= 'result', help = 'Result folder ')
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
    if args.model_type in ['seq2seq' , 'seq2seq_attn'] :
        exp_name = f"dataset_{args.dataset_name}/model_{args.model_type}_rcu_{args.rc_unit}_loss_{args.loss}_lr_{args.lr}_emb_{args.embed_size}_hs_{args.hidden_size}_stk_{args.rc_layers}"
    if args.model_type in ['transformer'] :
        exp_name = f"dataset_{args.dataset_name}/model_{args.model_type}_loss_{args.loss}_lr_{args.lr}_emb_{args.embed_size}"
    if args.model_type in ['t5_small', 't5', 'roberta', 'bart_large', 'bart_large_sum', 'pegasus', 'distill_bart'] :
        exp_name = f"dataset_{args.dataset_name}/model_{args.model_type}"

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