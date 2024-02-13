import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
import wandb, yaml
from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, test_iter


def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        warmup_steps=625
        gradient_clipping=1
        fc_dim=None
        batch_size=16
        loss_function = nn.CrossEntropyLoss()
        lr=0.0001

        train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                                batch_size=batch_size
        )


        model = TransformerClassifier(embed_dim=config.embed_dim, 
                                    num_heads=config.num_heads, 
                                    num_layers=config.num_layers,
                                    pos_enc=config.pos_enc,
                                    pool=config.pool,  
                                    dropout=config.dropout,
                                    fc_dim=fc_dim,
                                    max_seq_len=MAX_SEQ_LEN, 
                                    num_tokens=VOCAB_SIZE, 
                                    num_classes=NUM_CLS,
                                    )
        
        if torch.cuda.is_available():
            model = model.to('cuda')

        opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=config.weight_decay)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

        # create a folder to save the model
        if not os.path.exists('models'):
            os.makedirs('models')
        # training loop
        for e in range(config.num_epochs):
            print(f'\n epoch {e}')
            model.train()
            for batch in tqdm.tqdm(train_iter):
                opt.zero_grad()
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq)
                loss = loss_function(out, label) #compute loss
                loss.backward() # backward
                # if the total gradient vector has a length > 1, we clip it back down to 1.
                if gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                opt.step()
                sch.step()
                wandb.log({'train_loss': loss})

            #save validation accuracy and only save the model if it is the best one
            
            best_val_acc = 0
            with torch.no_grad():
                model.eval()
                tot, cor= 0.0, 0.0
                for batch in test_iter:
                    input_seq = batch.text[0]
                    batch_size, seq_len = input_seq.size()
                    label = batch.label - 1
                    if seq_len > MAX_SEQ_LEN:
                        input_seq = input_seq[:, :MAX_SEQ_LEN]
                    out = model(input_seq).argmax(dim=1)
                    tot += float(input_seq.size(0))
                    cor += float((label == out).sum().item())
                val_acc = np.round((cor / tot),3)
                print(f'-- {"validation"} accuracy {val_acc:.3}')
                wandb.log({'val_acc': val_acc, 'epoch': e})
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), 'models/' + wandb.run.name + '.pth')
                


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)

    with open('sweep_config.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)
    
    # print('Doing Bayesian Sweep to estimate best hyperparameters with respect to validation loss')
    # import pprint
    # pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="transformer-imdb1")
    
    wandb.agent(sweep_id,main,count=6)


    
