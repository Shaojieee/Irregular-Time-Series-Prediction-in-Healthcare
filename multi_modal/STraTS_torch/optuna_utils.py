import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from data import combine_values_varis, combine_values_varis_with_text, pad_text_data
from orig_model import STraTS as orig_STraTS
from strats_text_model import load_Bert

def objective(trial, train_dataset, val_dataset, accelerator, args):

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.001, log=True),
        'batch_size': trial.suggest_int('batch_size', 4, 64, 4),
        # 'embedding_size': trial.suggest_int('varis_dim', 4, args.d, 4),
        # 'TVE_hid_dim': trial.suggest_int('TVE_hid_dim', 4, args.d, 4),
        'N': trial.suggest_int('num_encoders', 1, 4, 1),
        'he': trial.suggest_int('num_heads', 2,8, 1),
        'd': trial.suggest_int('num_emb_dim', 32, 128, 4)
    }

    model = build_model(params, args)

    bce_loss = train_and_evaluate(params, trial, model, train_dataset, val_dataset, accelerator, args)

    return bce_loss



def build_model(params, args):
    if args.with_text:
        bert, bert_config, tokenizer = load_Bert(
            text_encoder_model = args.text_encoder_model
        )

        bert = accelerator.prepare(bert)
        
        model = orig_STraTS(
            D=D, # No. of static variables
            V=V+1, # No. of variables / features
            d=params['d'], # Input size of attention layer
            N=params['N'], # No. of Encoder blocks
            he=params['he'], # No. of heads in multi headed encoder blocks
            dropout=args.dropout,
            with_text=True,
            text_encoder=bert,
            text_encoder_name=args.text_encoder_model,
            text_linear_embed_dim=args.d*2,
            forecast=False, 
            return_embeddings=False,
            new_value_encoding=args.new_value_encoding,
            time_2_vec=args.time_2_vec
        )
    else:
        model = orig_STraTS(
            # No. of Demographics features
            D=args.D,
            # No. of Variable Embedding Size
            V=args.V,
            d=params['d'],
            N=params['N'],
            he=params['he'],
            dropout=args.dropout,
            forecast=False,
            return_embeddings=False,
            new_value_encoding=args.new_value_encoding,
            time_2_vec=args.time_2_vec
        )  
    
    print(model)

    return model



def train_and_evaluate(params, trial, model, train_dataset, val_dataset, accelerator, args):

    print(f'Parameters: {params}')

    if args.with_text and args.new_value_encoding:
        dataloader_collate_fn = lambda x: combine_values_varis_with_text(pad_text_data(x), normalise_varis=args.normalise_varis)
    elif args.with_text:
        dataloader_collate_fn = pad_text_data
    elif args.new_value_encoding:
        dataloader_collate_fn = lambda x: combine_values_varis(x, normalise_varis=args.normalise_varis)
    else:
        dataloader_collate_fn = None

    loss_fn = mortality_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], collate_fn=dataloader_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], collate_fn=dataloader_collate_fn)
    

    model, optimiser, train_dataloader, val_dataloader = \
    accelerator.prepare(model, optimiser, train_dataloader, val_dataloader)

    val_results = EvaluationCallback(
        val_dataloader=val_dataloader,
        evaluation_fn=mortality_results
    )

    early_stopper = EarlyStopper(
        patience=args.patience,
        min_delta=args.early_stopper_min_delta, 
        mode=args.early_stopper_mode, 
        restore_best_weights=False
    )


    for epoch in range(args.num_epochs):

        model.train()
        total_loss = 0.0
        for step, batch in tqdm(enumerate(train_dataloader)):

            if args.with_text:
                X_demos, X_times, X_values, X_varis, Y, X_text_tokens, X_text_attention_mask, X_text_times, X_text_time_mask, X_text_feature_varis = batch
                Y_pred = model(X_demos, X_times, X_values, X_varis, X_text_tokens, X_text_attention_mask, X_text_times, X_text_feature_varis)
            else:
                X_demos, X_times, X_values, X_varis, Y = batch
                Y_pred = model(X_demos, X_times, X_values, X_varis)
            

            loss = loss_fn(Y, Y_pred)

            accelerator.backward(loss)
            optimiser.step()
            optimiser.zero_grad()
            total_loss += loss.detach().cpu().item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Train Metrics: Epoch: {epoch} AVG LOSS: {avg_loss:.6f}')

        # Evaluation after 1 epoch
        results = val_results.on_epoch_end(
            model=model,
            epoch=epoch,
            with_text=args.with_text
        )

        
        trial.report(results['LOSS'], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        

        if early_stopper.on_epoch_end(model=model, loss=results['LOSS'], epoch=epoch):
            break
        
    
    return early_stopper.best
