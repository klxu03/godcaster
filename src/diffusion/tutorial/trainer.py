import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm
import warnings

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config
import os

import matplotlib.pyplot as plt

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'], heads=config["heads"])
    return model

def train_model(config):
    assert torch.cuda.is_available(), "You need a GPU to run this code"
    device = torch.device("cuda")

    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]


    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    losses = []

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            """
            if encoder_mask.dim() < 4: 
                encoder_mask = encoder_mask.unsqueeze(1)
            if encoder_mask.size(1) != config["heads"] and encoder_mask.size(1) == 1:
                encoder_mask = encoder_mask.repeat(1, config["heads"], 1, 1)
            elif encoder_mask.size(1) != config["heads"]:
                print("src_mask head dimension shape is weird, it is", encoder_mask.size(1))

            if decoder_mask.dim() < 4: 
                decoder_mask = decoder_mask.unsqueeze(1)
            if decoder_mask.size(1) != config["heads"] and decoder_mask.size(1) == 1:
                decoder_mask = decoder_mask.repeat(1, config["heads"], 1, 1)
            elif decoder_mask.size(1) != config["heads"]:
                print("tgt_mask head dimension shape is weird, it is", decoder_mask.size(1))
            """

            encoder_output = model.encode(encoder_input, encoder_mask) # output: (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)

            # (batch, seq_len, tgt_vocab_size) -> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            losses.append(loss.item())

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # Save model at end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

        print("current losses:", losses)

    # Plot losses
    plt.plot(losses)
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Loss over time")

    plt.savefig("loss_graph.png")
    plt.close()

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    pad_idx = tokenizer_tgt.token_to_id("[PAD]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # print("encoder source", source)
    # print("encoder_output", encoder_output)
    # Initialize the decoder input with the sos token
    """
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    padding_tokens = torch.full((1, max_len - 1), pad_idx, dtype=torch.long, device=device)
    decoder_input = torch.cat([decoder_input, padding_tokens], dim=1)
    """
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    curr_len = 1
    while True:
        if curr_len == max_len:
            break

        # build mask for target
        # decoder_mask = (decoder_input != pad_idx).unsqueeze(0).int() & causal_mask(decoder_input.size(0)).to(device) # (1, seq_len) broadcasted with & (1, seq_len, seq_len)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        print("decoder_mask shape", decoder_mask.size())

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        print(eos_idx, "next_word", next_word.item())

        # Replace the first padding token with the next word
        # decoder_input[0, curr_len] = next_word.item()
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        curr_len += 1

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            """
            if encoder_mask.dim() < 4: 
                encoder_mask = encoder_mask.unsqueeze(1)
            if encoder_mask.size(1) != config["heads"] and encoder_mask.size(1) == 1:
                encoder_mask = encoder_mask.repeat(1, config["heads"], 1, 1)
            elif encoder_mask.size(1) != config["heads"]:
                print("src_mask head dimension shape is weird, it is", encoder_mask.size(1))
            """

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            print("encoder_mask shape", encoder_mask.size())
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

def inference(config, input: str):
    assert torch.cuda.is_available(), "You need a GPU to run this code"
    device = torch.device("cuda")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    _, _, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    model_filename = get_weights_file_path(config, f"05")
    print(f"Preloading model {model_filename}")
    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])

    model.eval()

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    enc_input_tokens = tokenizer_src.encode(input).ids
    enc_num_padding_tokens = config["seq_len"] - len(enc_input_tokens) - 2
    sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ]
    )

    assert encoder_input.size(0) == config["seq_len"]

    encoder_mask = (encoder_input != pad_token).unsqueeze(1)
    encoder_mask = encoder_mask * encoder_mask.transpose(0, 1)
    encoder_mask = encoder_mask.unsqueeze(0)
    encoder_mask = encoder_mask.int()

    if encoder_mask.dim() < 4: 
        encoder_mask = encoder_mask.unsqueeze(1)
    if encoder_mask.size(1) != config["heads"] and encoder_mask.size(1) == 1:
        encoder_mask = encoder_mask.repeat(1, config["heads"], 1, 1)
    elif encoder_mask.size(1) != config["heads"]:
        print("src_mask head dimension shape is weird, it is", encoder_mask.size(1))

    print("encoder_mask shape", encoder_mask.size())

    with torch.no_grad():
        model_out = greedy_decode(model, encoder_input.unsqueeze(0).to(device), encoder_mask.to(device), tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        # Print the source, target and model output
        print('-'*console_width)
        print(f"{f'SOURCE: ':>12}{input}")
        print(f"{f'PREDICTED: ':>12}{model_out_text}")
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    # inference(config, "I love hot dog")
