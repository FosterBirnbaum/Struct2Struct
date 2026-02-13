import argparse
import os.path
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import queue
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess
import webdataset as wds
import esm as esmlib
import torch.distributed as dist
from coord_data import (WDSBatchSamplerWrapper, TERMLazyDataset, TERMBatchSamplerWrapper, TERMDataset, TERMLazyBatchSampler)
from coord_data_utils import DEFAULT_MODEL_HPARAMS, DEFAULT_TRAIN_HPARAMS, backwards_compat
from model_utils import loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def _setup_hparams(args):
    """ Setup the hparams dictionary using defaults and return it

    Args
    ----
    args : argparse.Namespace
        Parsed arguments

    Returns
    -------
    model_hparams : dict
        Fully configured model hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    run_hparams : dict
        Fully configured training run hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)
    """
    def _load_hparams(hparam_path, default_hparams, output_name):
        # load hparams
        hparams = json.load(open(hparam_path, 'r'))
        for key, default_val in default_hparams.items():
            if key not in hparams:
                hparams[key] = default_val

        hparams_path = os.path.join(args.path_for_outputs, output_name)
        if os.path.isfile(hparams_path):
            previous_hparams = json.load(open(hparams_path, 'r'))
            for key, default_val in default_hparams.items():
                if key not in previous_hparams:
                    previous_hparams[key] = default_val
            if previous_hparams != hparams:
                raise Exception('Given hyperparameters do not agree with previous hyperparameters.')
        else:
            json.dump(hparams, open(hparams_path, 'w'))

        return hparams

    model_hparams = _load_hparams(args.model_hparams, DEFAULT_MODEL_HPARAMS, 'model_hparams.json')
    run_hparams = _load_hparams(args.run_hparams, DEFAULT_TRAIN_HPARAMS, 'run_hparams.json')

    model_hparams, run_hparams = backwards_compat(model_hparams, run_hparams)

    return model_hparams, run_hparams

def _setup_dataloaders_wds(args, run_hparams, model_hparams):
    """ Setup dataloaders needed for training

    Args
    ----
    args : argparse.Namespace
        Parsed arguments
    run_hparams : dict
        Fully configured hparams dictionary (see :code:`terminator/utils/model/default_hparams.py`)

    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader : torch.utils.data.DataLoader
        DataLoaders for the train, validation, and test datasets
    """

    # set up dataloaders
    # train_ids = []
    # with open(args.train, 'r') as f:
    #     for line in f:
    #         train_ids += [line[:-1]]
    # validation_ids = []
    # with open(args.validation, 'r') as f:
    #     for line in f:
    #         validation_ids += [line[:-1]]
    # test_ids = []
    # with open(args.test, 'r') as f:
    #     for line in f:
    #         test_ids += [line[:-1]]
    dataset = args.path_for_training_data
    if run_hparams['msa_type']:
        dataset = args.msa_dataset

    # train_dataset = TERMDataset(dataset, pdb_ids=train_ids)
    # val_dataset = TERMDataset(dataset, pdb_ids=validation_ids)
    # test_dataset = TERMDataset(args.dataset, pdb_ids=test_ids)
    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_dataset))
    pair_etab_dir = ''
    if 'edge_loss_msa' in run_hparams['loss_config'].keys() or 'msa_weighted_nlcpl' in run_hparams['loss_config'].keys():
        pair_etab_dir = dataset
    cols = ['inp.pyd']
    
    if model_hparams['esm_feats']:
        if model_hparams['esm_model'] == '650':
            esm, alphabet = esmlib.pretrained.esm2_t33_650M_UR50D()
        else:
            esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        esm = esm.eval().cpu()
    else:
        esm, batch_converter = None, None
    
    train_dataset = wds.WebDataset(os.path.join(args.path_for_training_data, 'multichain_train.wds')).decode().to_tuple(*cols)
    val_dataset = wds.WebDataset(os.path.join(args.path_for_training_data, 'multichain_val.wds')).decode().to_tuple(*cols)
    test_dataset = wds.WebDataset(os.path.join(args.path_for_training_data, 'multichain_test.wds')).decode().to_tuple(*cols)
    print('train dataset: ', os.path.join(args.path_for_training_data, 'multichain_train.wds'))
    train_dataset = [data for data in train_dataset if (data[0]['coords'].shape[0] == data[0]['sc_coords'].shape[0]) and (data[0]['coords'].shape[0] > 30)]
    val_dataset = [data for data in val_dataset if (data[0]['coords'].shape[0] == data[0]['sc_coords'].shape[0]) and (data[0]['coords'].shape[0] > 30)]
    test_dataset = [data for data in test_dataset if (data[0]['coords'].shape[0] == data[0]['sc_coords'].shape[0]) and (data[0]['coords'].shape[0] > 30)]

    if model_hparams['esm_feats'] and run_hparams['use_esm_attns']:
        train_dataset = [data for data in train_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence'])) and (data[0]['coords'].shape[0] < 2000)]
        val_dataset = [data for data in val_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence'])) and (data[0]['coords'].shape[0] < 2000)]
        test_dataset = [data for data in test_dataset if (data[0]['coords'].shape[0] == len(data[0]['sequence'])) and (data[0]['coords'].shape[0] < 2000)]
    print('train len: ', len(train_dataset))
    print('val len: ', len(val_dataset))
    print('test len: ', len(test_dataset))
    train_batch_sampler = WDSBatchSamplerWrapper(dist.is_initialized())
    train_batch_sampler = train_batch_sampler.sampler(train_batch_sampler.ddp, train_dataset,
                                            batch_size=run_hparams['train_batch_size'],
                                            shuffle=run_hparams['shuffle'],
                                            semi_shuffle=run_hparams['semi_shuffle'],
                                            sort_data=run_hparams['sort_data'],
                                            max_term_res=run_hparams['max_term_res'],
                                            max_seq_tokens=run_hparams['max_seq_tokens'],
                                            msa_type=run_hparams['msa_type'],
                                            msa_id_cutoff=run_hparams['msa_id_cutoff'],
                                            flex_type=run_hparams['flex_type'],
                                            replicate=run_hparams['replicate'],
                                            noise_level=run_hparams['noise_level'],
                                            bond_length_noise_level=run_hparams['bond_length_noise_level'],
                                            bond_angle_noise_level=run_hparams['bond_angle_noise_level'],
                                            noise_lim=run_hparams['noise_lim'],
                                            pair_etab_dir=pair_etab_dir,
                                            use_sc=run_hparams["use_sc"], 
                                            sc_mask_rate=run_hparams['sc_mask_rate'],
                                            base_sc_mask=run_hparams['base_sc_mask'],
                                            sc_mask_schedule=run_hparams['sc_mask_schedule'],
                                            sc_info=run_hparams['sc_info'],
                                            sc_noise=run_hparams['sc_noise'],
                                            mask_neighbors=run_hparams['mask_neighbors'],
                                            mask_interface=run_hparams['mask_interface'],
                                            half_interface=run_hparams['half_interface'],
                                            inter_cutoff=run_hparams['inter_cutoff'],
                                            dev=args.dev,
                                            esm=esm,
                                            batch_converter=batch_converter,
                                            use_esm_attns=run_hparams['use_esm_attns'],
                                            use_reps=run_hparams['use_reps'],
                                            post_esm_mask=run_hparams['post_esm_mask'],
                                            from_rla=model_hparams['rla_feats'],
                                            esm_embed_layer=model_hparams['esm_embed_layer'],
                                            connect_chains=run_hparams['connect_chains'],
                                            convert_to_esm=model_hparams['convert_to_esm'],
                                            one_hot=model_hparams['one_hot'])
    val_batch_sampler = WDSBatchSamplerWrapper(dist.is_initialized())
    val_batch_sampler = val_batch_sampler.sampler(val_batch_sampler.ddp, val_dataset, batch_size=1, shuffle=False, msa_type=run_hparams['msa_type'], msa_id_cutoff=run_hparams['msa_id_cutoff'], pair_etab_dir=pair_etab_dir, use_sc=run_hparams["use_sc"], sc_mask_rate=run_hparams['sc_mask_rate'], base_sc_mask=run_hparams['base_sc_mask'], sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], inter_cutoff=run_hparams['inter_cutoff'], dev=args.dev, esm=esm, batch_converter=batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'])

    test_batch_sampler = WDSBatchSamplerWrapper(dist.is_initialized())
    
    test_batch_sampler = test_batch_sampler.sampler(test_batch_sampler.ddp, test_dataset, batch_size=1, shuffle=False, use_sc=run_hparams["use_sc"], sc_mask_rate=run_hparams['sc_mask_rate'], base_sc_mask=run_hparams['base_sc_mask'], sc_mask_schedule=run_hparams['sc_mask_schedule'], sc_info=run_hparams['sc_info'], mask_neighbors=run_hparams['mask_neighbors'], mask_interface=run_hparams['mask_interface'], half_interface=run_hparams['half_interface'], inter_cutoff=run_hparams['inter_cutoff'], dev=args.dev, esm=esm, batch_converter=batch_converter, use_esm_attns=run_hparams['use_esm_attns'], use_reps=run_hparams['use_reps'], post_esm_mask=run_hparams['post_esm_mask'], from_rla=model_hparams['rla_feats'], esm_embed_layer=model_hparams['esm_embed_layer'], connect_chains=run_hparams['connect_chains'], convert_to_esm=model_hparams['convert_to_esm'], one_hot=model_hparams['one_hot'])
    
    return train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler


def main(args):

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    args.dev = device

    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint

    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    data_path = args.path_for_training_data


    print('loaded loaders')
    kwargs = {}
    kwargs['num_workers'] = 20
    model_hparams, run_hparams = _setup_hparams(args)
    train_dataset, train_batch_sampler, val_dataset, val_batch_sampler, test_dataset, test_batch_sampler = _setup_dataloaders_wds(args, run_hparams, model_hparams)
    loader_train = DataLoader(train_dataset,
                                batch_sampler=train_batch_sampler,
                                collate_fn=train_batch_sampler.package,
                                pin_memory=True,
                                **kwargs)
    loader_valid = DataLoader(val_dataset,
                                batch_sampler=val_batch_sampler,
                                collate_fn=val_batch_sampler.package,
                                pin_memory=True,
                                **kwargs)
    # test_dataloader = DataLoader(test_dataset,
    #                             batch_sampler=test_batch_sampler,
    #                             collate_fn=test_batch_sampler.package,
    #                             **kwargs)

    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise,
                        augment_type=args.noise_type,
                        feat_type=args.feat_type)
    model.to(device)

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)


    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if run_hparams['finetune']:
        for (name, module) in model.named_children():
            print(name)
            if "W_out" in name:
                print('\tSet grad to true')
                module.requires_grad = True
            else:
                module.requiers_grad = False
                    

    print('setup model')
    kwargs = {}
    kwargs['num_workers'] = 32

    for e in range(args.num_epochs):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        train_acc = 0.

        for _, batch in enumerate(loader_train):
            start_batch = time.time()
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, all_chain_lens = batch
            residue_idx = residue_idx.to(device=device)
            S = S.to(device=device)
            X = X.to(device=device)
            mask = mask.to(device=device)
            mask_self = mask_self.to(device=device)
            chain_M = chain_M.to(device=device)
            chain_encoding_all =chain_encoding_all.to(device=device)
            elapsed_featurize = time.time() - start_batch
            optimizer.zero_grad()
            mask_for_loss = mask*chain_M
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, all_chain_lens, e, replicate=args.replicate)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
        
                scaler.scale(loss_av_smoothed).backward()
                    
                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, all_chain_lens, e, replicate=args.replicate)
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()
            
            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
        
            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            total_step += 1

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            validation_acc = 0.
            for _, batch in enumerate(loader_valid):
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, all_chain_lens = batch
                residue_idx = residue_idx.to(device=device)
                S = S.to(device=device)
                X = X.to(device=device)
                mask = mask.to(device=device)
                mask_self = mask_self.to(device=device)
                chain_M = chain_M.to(device=device)
                chain_encoding_all =chain_encoding_all.to(device=device)
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                mask_for_loss = mask*chain_M
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                
                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
        
        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)
        
        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
        with open(logfile, 'a') as f:
            f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
        print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
        
        checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
        torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'noise_type': args.noise_type,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename_last)

        if (e+1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
            torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'noise_type': args.noise_type, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    argparser.add_argument("--noise_type", type=str, default='atomic', help="type of noise to add during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--replicate", type=int, default=1, help='replicate to use in setting seed for noise')
    argparser.add_argument("--feat_type", type=str, default="protein_mpnn", help="type of featurizer to use")
    argparser.add_argument("--model_hparams", help='file path for model hparams', required=True)
    argparser.add_argument('--run_hparams', help='file path for run hparams', required=True)
 
    args = argparser.parse_args()
    print('starting')    
    main(args)   
