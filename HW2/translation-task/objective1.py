import torch
import pandas as pd
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import constants
from nltk.translate.bleu_score import corpus_bleu
from dataload import Multi30kDataset, make_collator
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists("Weights"):
    os.mkdir("Weights")
if not os.path.exists("Resulted_Models"):
    os.mkdir("Resulted_Models")

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout_prob, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, embedding_size).to(self.device)
        self.LSTM = torch.nn.LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, device=self.device)
        self.dropout = torch.nn.Dropout(dropout_prob).to(self.device)

    def forward(self, x):
        embeddings = self.embedding(x.to(self.device))
        embeddings = self.dropout(embeddings)
        output, hidden = self.LSTM(embeddings)
        return output, hidden

class DecoderRNN(torch.nn.Module):
    def __init__(self, decoder_input_size, embedding_size, hidden_size, sos_token, eos_token, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(decoder_input_size, embedding_size, device=self.device)
        self.LSTM = torch.nn.LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, device=self.device)
        self.linear = torch.nn.Linear(hidden_size, decoder_input_size, device=self.device)

    def generate(self, encoder_outputs, encoder_hidden, max_len):
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.sos_token)
        decoder_hidden = encoder_hidden

        generated_sentences = []
        for _ in range(max_len):
            out, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            
            _, top_index = out.topk(1)
            decoder_input = top_index.squeeze(-1)
            
            generated_sentences.append(decoder_input)

            if (decoder_input == self.eos_token).all():
                break

        generated_sentences = torch.stack(generated_sentences, dim=1)
        return generated_sentences

    def forward(self, encoder_outputs, encoder_hidden, targets=None, teacher_forcing_p=0.0):
        decoder_outputs = []
        batch_size = encoder_outputs.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.sos_token)
        decoder_hidden = encoder_hidden

        num_steps = targets.shape[1] if targets is not None else encoder_outputs.shape[1]

        for i in range(num_steps):
            out, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(out)

            if i>0 and torch.rand(1).item() < teacher_forcing_p:
                decoder_input = targets[:,i].unsqueeze(1).to(self.device)
            else:
                _, topindex = out.squeeze(1).topk(1, dim=-1)
                decoder_input = topindex.detach()
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.nn.functional.log_softmax(decoder_outputs, dim=-1)

        return decoder_outputs, decoder_hidden

    def forward_step(self, input, hidden):
        embeddings = self.embedding(input.to(self.device))
        output, hidden = self.LSTM(embeddings, hidden)
        output = self.linear(output)
        return output, hidden

def evaluate_bleu(encoder, decoder, valid_dataloader, max_len=50):
    all_generated = []
    all_references = []

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for idx, data in tqdm(enumerate(valid_dataloader, 0)):
            input_tensor, target_tensor = data 
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            generated = decoder.generate(encoder_outputs, encoder_hidden, max_len)
            
            for i in range(generated.size(0)):
                generated_sentence = [generated[i, j].item() for j in range(generated.size(1)) 
                                   if generated[i, j].item() != decoder.eos_token]
                all_generated.append(generated_sentence)

                reference_sentence = [target_tensor[i, j].item() for j in range(target_tensor.size(1)) 
                                   if target_tensor[i, j].item() != decoder.eos_token]
                all_references.append([reference_sentence])

    bleu_score = corpus_bleu(all_references, all_generated)
    print(f"BLEU Score: {bleu_score}")
    return bleu_score

def evaluate_perplexity(encoder, decoder, valid_dataloader):
    total_loss = 0
    total_tokens = 0
    decoder.eval()
    criterion = torch.nn.NLLLoss()

    with torch.no_grad():  
        for idx, data in tqdm(enumerate(valid_dataloader, 0)):
            input_tensor, target_tensor = data 
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, targets=target_tensor)

            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
            total_loss += loss.item() * target_tensor.size(0)
            total_tokens += target_tensor.size(0)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    print(f"Perplexity: {perplexity.item()}")
    return perplexity.item()

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_p):
    total_loss = 0
    encoder.train()
    decoder.train()

    for idx, data in tqdm(enumerate(dataloader, 0)):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_tensor, teacher_forcing_p)

        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(encoder, decoder, train_dataloader, valid_dataloader, n_epochs, learning_rate, 
          enc_weights_path, dec_weights_path, teacher_forcing_p):
    train_losses = []
    valid_losses = []  
    valid_bleu_scores = []
    valid_perplexities = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        print("Start epoch ", epoch)
        
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, 
                          decoder_optimizer, criterion, teacher_forcing_p)
        print(f'Train Loss: {train_loss:.4f}')
        train_losses.append(train_loss)

        valid_loss = 0
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for data in valid_dataloader:
                input_tensor, target_tensor = data
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                encoder_outputs, encoder_hidden = encoder(input_tensor)
                decoder_outputs, _ = decoder(encoder_outputs, encoder_hidden, target_tensor, 0)
                
                loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
                valid_loss += loss.item()
        
        valid_loss /= len(valid_dataloader)
        valid_losses.append(valid_loss)
        print(f'Validation Loss: {valid_loss:.4f}')

        bleu_score = evaluate_bleu(encoder, decoder, valid_dataloader)
        perplexity = evaluate_perplexity(encoder, decoder, valid_dataloader)
        
        valid_bleu_scores.append(bleu_score)
        valid_perplexities.append(perplexity)

        torch.save(encoder.state_dict(), enc_weights_path)
        torch.save(decoder.state_dict(), dec_weights_path)         

    return encoder, decoder, train_losses, valid_losses, valid_bleu_scores, valid_perplexities

def plot_comparison_metrics(test_name, configs, all_metrics):
    if test_name == 'dropout_test':
        varied_param = 'dropout'
    elif test_name == 'embedding_test':
        varied_param = 'emb_dim'
    elif test_name == 'hidden_dim_test':
        varied_param = 'hid_dim'
    elif test_name == 'batch_size_test':
        varied_param = 'batch_size'
    else: 
        varied_param = 'tf_ratio'
    
    # Training Loss
    plt.figure(figsize=(10, 6))
    for i, params in enumerate(configs):
        param_value = params[varied_param]
        train_losses = all_metrics[i]['train_losses']
        plt.plot(train_losses, label=f'{varied_param}={param_value}')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Results/training_loss_{test_name}.png', bbox_inches='tight')
    plt.close()

    # Validation Loss
    plt.figure(figsize=(10, 6))
    for i, params in enumerate(configs):
        param_value = params[varied_param]
        valid_losses = all_metrics[i]['valid_losses']
        plt.plot(valid_losses, label=f'{varied_param}={param_value}')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Results/validation_loss_{test_name}.png', bbox_inches='tight')
    plt.close()

    # BLEU Score
    plt.figure(figsize=(10, 6))
    for i, params in enumerate(configs):
        param_value = params[varied_param]
        bleu_scores = all_metrics[i]['valid_bleu_scores']
        plt.plot(bleu_scores, label=f'{varied_param}={param_value}')
    plt.title('BLEU Score Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Results/bleu_score_{test_name}.png', bbox_inches='tight')
    plt.close()

    # Perplexity
    plt.figure(figsize=(10, 6))
    for i, params in enumerate(configs):
        param_value = params[varied_param]
        perplexities = all_metrics[i]['valid_perplexities']
        plt.plot(perplexities, label=f'{varied_param}={param_value}')
    plt.title('Perplexity Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Results/perplexity_{test_name}.png', bbox_inches='tight')
    plt.close()

def main():
    df_train = pd.read_parquet("data/tokenized/train.parquet")
    df_valid = pd.read_parquet("data/tokenized/valid.parquet")

    with open("models/vocab_en.pkl", "rb") as _file:
        en_vocab = pickle.load(_file)
    with open("models/vocab_fr.pkl", "rb") as _file:
        fr_vocab = pickle.load(_file)

    set_train = Multi30kDataset(df_train, en_vocab, fr_vocab)
    set_valid = Multi30kDataset(df_valid, en_vocab, fr_vocab)

    fr_SOS_token = fr_vocab.token_to_index(constants.SOS)
    fr_EOS_token = fr_vocab.token_to_index(constants.EOS)

    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3

    configs = {
        'dropout_test': [
            {'emb_dim': 128, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.1, 'tf_ratio': 0.5},
            {'emb_dim': 128, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5}
        ],
        'embedding_test': [
            {'emb_dim': 128, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5},
            {'emb_dim': 256, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5},
            {'emb_dim': 512, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5}
        ],
        'hidden_dim_test': [
            {'emb_dim': 256, 'hid_dim': 128, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5},
            {'emb_dim': 256, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5},
            {'emb_dim': 256, 'hid_dim': 512, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5}
        ],
        'batch_size_test': [
            {'emb_dim': 256, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5},
            {'emb_dim': 256, 'hid_dim': 256, 'batch_size': 256, 'dropout': 0.5, 'tf_ratio': 0.5}
        ],
        'teacher_forcing_test': [
            {'emb_dim': 256, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.0},
            {'emb_dim': 256, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 0.5},
            {'emb_dim': 256, 'hid_dim': 256, 'batch_size': 128, 'dropout': 0.5, 'tf_ratio': 1.0}
        ]
    }

    if not os.path.exists("Results"):
        os.makedirs("Results")

    for test_name, test_configs in configs.items():
        print(f"\nRunning {test_name}")
        all_metrics = []
        
        test_results = pd.DataFrame(columns=[
            'config_id', 'varied_param', 'param_value', 'final_bleu', 
            'final_perplexity', 'best_bleu', 'best_epoch'
        ])

        if test_name == 'dropout_test':
            varied_param = 'dropout'
        elif test_name == 'embedding_test':
            varied_param = 'emb_dim'
        elif test_name == 'hidden_dim_test':
            varied_param = 'hid_dim'
        elif test_name == 'batch_size_test':
            varied_param = 'batch_size'
        else:
            varied_param = 'tf_ratio'

        for idx, config in enumerate(test_configs):
            print(f"\nTraining model {idx + 1}/{len(test_configs)} for {test_name}:")
            print(config)

            model_params = f'{test_name}_E_{config["emb_dim"]}_H_{config["hid_dim"]}_B_{config["batch_size"]}_D_{config["dropout"]}_T_{config["tf_ratio"]}'

            collator = make_collator(en_vocab, fr_vocab)
            train_dataloader = DataLoader(set_train, collate_fn=collator, batch_size=config['batch_size'])
            valid_dataloader = DataLoader(set_valid, collate_fn=collator, batch_size=config['batch_size'])

            encoder_weights_path = f'Weights/encoder_{model_params}.pt'
            decoder_weights_path = f'Weights/decoder_{model_params}.pt'

            encoder = EncoderRNN(
                input_size=len(en_vocab),
                embedding_size=config['emb_dim'],
                hidden_size=config['hid_dim'],
                dropout_prob=config['dropout'],
                device=device
            )
            decoder = DecoderRNN(
                decoder_input_size=len(fr_vocab),
                embedding_size=config['emb_dim'],
                hidden_size=config['hid_dim'],
                sos_token=fr_SOS_token,
                eos_token=fr_EOS_token,
                device=device
            )
            
            encoder = encoder.to(device)
            decoder = decoder.to(device)

            encoder, decoder, train_losses, valid_losses, valid_bleu_scores, valid_perplexities = train(
                encoder, decoder,
                train_dataloader,
                valid_dataloader,
                n_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                enc_weights_path=encoder_weights_path,
                dec_weights_path=decoder_weights_path,
                teacher_forcing_p=config['tf_ratio']
            )

            all_metrics.append({
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'valid_bleu_scores': valid_bleu_scores,
                'valid_perplexities': valid_perplexities
            })

            new_row = pd.DataFrame({
                'config_id': [idx + 1],
                'varied_param': [varied_param],
                'param_value': [config[varied_param]],
                'final_bleu': [valid_bleu_scores[-1]],
                'final_perplexity': [valid_perplexities[-1]],
                'best_bleu': [max(valid_bleu_scores)],
                'best_epoch': [np.argmax(valid_bleu_scores) + 1]
            })
            test_results = pd.concat([test_results, new_row], ignore_index=True)

        test_results.to_csv(f'Results/results_{test_name}.csv', index=False)
        
        plot_comparison_metrics(test_name, test_configs, all_metrics)

if __name__ == "__main__":
    main()