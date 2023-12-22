from utils import *
from data import *
from model import STraTS
from strats_text_model import STraTS_text, load_Bert

import warnings
import time
import logging
logger = logging.getLogger(__name__)
from datetime import datetime
import os
from tqdm import tqdm
import torch
import torch.nn
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator



def main():
    args = parse_args()
    args.start_time = datetime.now()
    if args.output_dir==None:
        args.output_dir = './logs/' + args.start_time.strftime('%Y%m%d_%H%M')
    os.makedirs(args.output_dir, exist_ok=True)

    print(args)
    if args.fp16:
        args.mixed_precision="fp16"
    else:
        args.mixed_precision="no"
    accelerator = Accelerator(fp16=args.fp16, mixed_precision=args.mixed_precision,cpu=args.cpu)

    device = accelerator.device
    print(f'Device:{device}')

    if args.train_job=='forecasting_model':
        train_forecasting_model(args, accelerator)
    elif args.train_job=='mortality_model':
        train_mortality_model(args, accelerator)



def train_forecasting_model(args, accelerator):
    train_dataset, val_dataset, fore_max_len, V, D = load_forecast_dataset(args.data_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size)

    model = STraTS(
        # No. of Demographics features
        D=D,
        # Input size of the  Variable Embedding
        V=V,
        # Output size of the embedding vector
        d=args.d,
        # No. of Encoder blocks (Contextual Triplet Embedding)
        N=args.N,
        # No. of attention heads in Encoder blocks (Contextual Triplet Embedding)
        he=args.he,
        dropout=args.dropout,
        forecast=True
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_fn = forecast_loss
    model, optimiser, train_dataloader,val_dataloader = accelerator.prepare(model, optimiser, train_dataloader,val_dataloader)

    early_stopper = EarlyStopper(
        patience=args.patience,
        min_delta=args.early_stopper_min_delta, 
        mode=args.early_stopper_mode, 
        restore_best_weights=args.early_stopper_restore_best_weights
    )

    eval_results = EvaluationCallback(
        val_dataloader = val_dataloader,
        evaluation_fn=forecast_results
    )


    # TODO: Include training results log file as well
    for epoch in tqdm(range(args.num_epochs)):
        if time_check(args.start_time):
            break

        model.train()
        total_loss = 0.0
        for step, batch in tqdm(enumerate(train_dataloader)):
            # Restricting the number of samples per training epoch
            if args.samples_per_epoch!=None and args.samples_per_epoch<step*args.train_batch_size:
                break
            
            X_demos, X_times, X_values, X_varis, Y = batch

            Y_pred = model(X_demos, X_times, X_values, X_varis)
            loss = loss_fn(Y, Y_pred, V)
            accelerator.backward(loss)
            optimiser.step()
            optimiser.zero_grad()
            total_loss += loss.detach().cpu().item()
        print(f'Train Metrics: Epoch: {epoch} LOSS: {total_loss:.6f}')
        
        model.eval()
        # Evaluation after 1 epoch
        results = eval_results.on_epoch_end(
            model=model,
            epoch=epoch,
            V=V
        )

        # Early Stopper check
        if early_stopper.on_epoch_end(model=model, loss=results['LOSS'], epoch=epoch):
            break
    
    # End of training
    model.eval()
    # Saving Weights
    early_stopper.save_best_weights(
        accelerator=accelerator,
        output_dir=args.output_dir,
        file_name='best_forecasting_model.pth'
    )
    # Saving Experiment Details
    save_experiment_config(
        args=args,
        output_dir=args.output_dir,
        file_name=f'{args.train_job}_details.json'
    )
    # Saving Evaluation Logs
    eval_results.save_results(
        output_dir=args.output_dir,
        file_name=f'{args.train_job}_val_results.csv'
    )
    

def train_mortality_model(args, accelerator):

    if args.with_text:
        _, _, tokenizer = load_Bert(
            text_encoder_model = args.text_encoder_model
        )
    else:
        tokenizer = None
        dataloader_collate_fn = None
    
    if args.with_text and args.new_value_encoding:
        dataloader_collate_fn = lambda x: combine_values_varis_with_text(pad_text_data(x))
    elif args.with_text:
        dataloader_collate_fn = pad_text_data
    elif args.new_value_encodingL
        dataloader_collate_fn = combine_values_varis
    else:
        dataloader_collate_fn = None

    train_dataset, val_dataset, test_dataset, V, D = load_mortality_dataset(
        args.data_dir, 
        with_text=args.with_text, 
        tokenizer=tokenizer,
        text_padding=args.text_padding, 
        text_max_len=args.text_max_length, 
        text_model=args.text_encoder_model, 
        period_length=args.period_length, 
        num_notes=args.text_num_notes,
        debug=False
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=dataloader_collate_fn)

    print(f'Training Dataset: {len(train_dataset)}')
    print(f'Validation Dataset: {len(val_dataset)}')
    print(f'Test Dataset: {len(test_dataset)}')

    print_model = True
    for ld in args.lds:
        
        if time_check(args.start_time):
            break
        # Generating list of start indexes for different repeats
        train_start = [int(i) for i in np.linspace(0, len(train_dataset)-int(ld*len(train_dataset)/100), args.repeats)]
        val_start = [int(i) for i in np.linspace(0, len(val_dataset)-int(ld*len(val_dataset)/100), args.repeats)]
        
        for i in range(args.repeats):
            print(f'Training with ld: {ld} Repeat {i+1}')
            if time_check(args.start_time):
                break
            # Generating list of indexes
            cur_train_ind = np.arange(train_start[i], train_start[i]+int(ld*len(train_dataset)/100))
            train_subset = Subset(train_dataset, cur_train_ind)

            cur_val_ind = np.arange(val_start[i], val_start[i]+int(ld*len(val_dataset)/100))
            val_subset = Subset(val_dataset, cur_val_ind)

            train_dataloader = DataLoader(train_subset, batch_size=args.train_batch_size, collate_fn=dataloader_collate_fn)
            val_dataloader = DataLoader(val_subset, batch_size=args.eval_batch_size, collate_fn=dataloader_collate_fn)

            # TODO: with text model
            if args.with_text:
                bert, bert_config, tokenizer = load_Bert(
                    text_encoder_model = args.text_encoder_model
                )

                bert = accelerator.prepare(bert)
                
                model = STraTS(
                    D=D, # No. of static variables
                    V=V+1, # No. of variables / features
                    d=args.d, # Input size of attention layer
                    N=args.N, # No. of Encoder blocks
                    he=args.he, # No. of heads in multi headed encoder blocks
                    dropout=args.dropout,
                    with_text=True,
                    text_encoder=bert,
                    text_encoder_name=args.text_encoder_model,
                    text_linear_embed_dim=args.d*2,
                    forecast=False, 
                    return_embeddings=False
                )
            else:
                model = STraTS(
                    # No. of Demographics features
                    D=D,
                    # No. of Variable Embedding Size
                    V=V,
                    d=args.d,
                    N=args.N,
                    he=args.he,
                    dropout=args.dropout,
                    forecast=False
                )
            

            if args.model_weights!=None:
                model_weights = torch.load(args.model_weights, map_location=torch.device('cpu'))
                try:
                    model.load_state_dict(model_weights, strict=True)
                except Exception as e:
                    print('Model do no match saved weights')
                    print(e)

                    model.load_state_dict(model_weights, strict=False)
            
            

            optimiser = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
            if args.with_text:
                # optimizer= torch.optim.Adam([
                #     {'params': [p for n, p in model.named_parameters() if 'text' not in n]},
                #     {'params':[p for n, p in model.named_parameters() if 'text' in n], 'lr': args.text_learning_rate}
                # ], lr=args.ts_learning_rate)

                for param in model.text_encoder.parameters():
                        param.requires_grad = False
            
            if print_model:
                print(model)
                total_params = sum(p.numel() for p in model.parameters())
                total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
                print(f'Total number of parameters: {total_params}, Total trainable parameters: {total_trainable_params}')
                print_model = False

            loss_fn = mortality_loss

            model, optimiser, train_dataloader, val_dataloader, test_dataloader = \
            accelerator.prepare(model, optimiser, train_dataloader, val_dataloader, test_dataloader)

            early_stopper = EarlyStopper(
                patience=args.patience,
                min_delta=args.early_stopper_min_delta, 
                mode=args.early_stopper_mode, 
                restore_best_weights=args.early_stopper_restore_best_weights
            )

            val_results = EvaluationCallback(
                val_dataloader=val_dataloader,
                evaluation_fn=mortality_results
            )

            test_results = EvaluationCallback(
                val_dataloader=test_dataloader,
                evaluation_fn=mortality_results
            )


            # TODO: Include training results log file as well
            for epoch in tqdm(range(args.num_epochs)):
                
                if time_check(args.start_time, mins_left=40):
                    break

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

                # Early Stopper check
                if early_stopper.on_epoch_end(model=model, loss=results['LOSS'], epoch=epoch):
                    break
    
            # End of training

            # Restore best weights
            early_stopper.load_best_weights(
                model=model
            )

            # Getting Test Results
            test_results.on_epoch_end(
                model=model,
                epoch=epoch,
                with_text=args.with_text
            )

            # Saving Weights
            early_stopper.save_best_weights(
                accelerator=accelerator,
                output_dir=args.output_dir,
                file_name=f'best_mortality_model_{ld}_repeat_{i}.pth'
            )
            
            # Saving Validation Logs
            val_results.save_results(
                output_dir=args.output_dir,
                file_name=f'{args.train_job}_val_results_{ld}_repeat_{i}.csv'
            )

            # Saving Test Logs
            test_results.save_results(
                output_dir=args.output_dir,
                file_name=f'{args.train_job}_test_results_{ld}_repeat_{i}.csv'
            )

            accelerator.free_memory()


    # Saving Experiment Details
    save_experiment_config(
        args=args,
        output_dir=args.output_dir,
        file_name=f'{args.train_job}_details.json'
    )


if __name__ == "__main__":
    main()