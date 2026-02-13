import argparse
import os.path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main(args):
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
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor 
    import esm as esmlib
    from utils2 import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils2 import featurize, loss_smoothed, nlcpl, structure_loss, loss_nll, get_std_opt, ProteinMPNN

    scaler = torch.cuda.amp.GradScaler()
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

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
    data_meta_path = args.path_for_meta_data
    if len(args.consensus_seqs) > 0 or args.msa_seqs:
        consensus_tag = '_consensus'
    else:
        consensus_tag = ''
    params = {
        "LIST"    : f"{data_meta_path}/list{consensus_tag}.csv", 
        "VAL"     : f"{data_meta_path}/valid_clusters.txt",
        "TEST"    : f"{data_meta_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70, #min seq.id. to detect homo chains
        "CATH"    : "ingraham" in data_path
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 2}

   
    if args.debug:
        args.num_examples_per_epoch = 500
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)

    print('train len / val len: ', len(train), len(valid))
     
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    use_potts = args.etab_loss or args.etab_loss_only
    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise,
                        use_potts=use_potts,
                        node_self_sub=args.node_self_sub,
                        clone=args.clone,
                        seq_encoding=args.seq_encoding,
                        output_dim=args.output_dim,
                        )
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


    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for i in range(3):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()
       
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
        
        loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
        
        reload_c = 0 
        best_val_loss = np.inf
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            nlcpl_train_sum = 0.
            train_acc = 0.
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=args.batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, args.max_protein_length, args.num_examples_per_epoch))
                reload_c += 1

            for i_train_batch, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        log_probs, etab, E_idx, frames, positions = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                        if args.etab_loss:
                            nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S, mask)
                            if nlcpl_count >= 0: loss_av_smoothed += nlcpl_loss
                            else:
                                raise ValueError
                        elif args.etab_loss_only:
                            loss_av_smoothed, nlcpl_count = nlcpl(etab, E_idx, S, mask)
                    scaler.scale(loss_av_smoothed).backward()
                     
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs, etab, E_idx, frames, positions = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    if args.etab_loss:
                            nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S, mask)
                            if nlcpl_count >= 0: loss_av_smoothed += nlcpl_loss
                            else:
                                raise ValueError
                    elif args.etab_loss_only:
                            loss_av_smoothed, nlcpl_count = nlcpl(etab, E_idx, S, mask)
                    loss_av_smoothed.backward()

                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                if args.etab_loss or args.etab_loss_only:
                    # nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S_true, mask)
                    if nlcpl_count >= 0:
                        nlcpl_train_sum += nlcpl_loss.cpu().item()
                    else:
                        nlcpl_train_sum += np.mean(nlcpl_train_sum)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                nlcpl_validation_sum = 0.
                for i_val_batch, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs, etab, E_idx, frames, positions = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    if args.etab_loss or args.etab_loss_only:
                        nlcpl_loss, nlcpl_count = nlcpl(etab, E_idx, S, mask)
                        if nlcpl_count >= 0:
                            nlcpl_validation_sum += nlcpl_loss.cpu().item()
                        else:
                            nlcpl_validation_sum += np.mean(nlcpl_validation_sum)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            train_loss = train_sum / train_weights
            train_comb_loss = copy.deepcopy(train_loss)
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
            comb_loss = copy.deepcopy(validation_loss)

            if args.etab_loss or args.etab_loss_only:
                train_nlcpl = nlcpl_train_sum / i_train_batch
                validation_nlcpl = nlcpl_validation_sum / i_val_batch
                train_comb_loss += train_nlcpl
                comb_loss += validation_nlcpl
                train_nlcpl = np.format_float_positional(np.float32(train_nlcpl), unique=False, precision=3)   
                validation_nlcpl = np.format_float_positional(np.float32(validation_nlcpl), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train_loss: {train_comb_loss}, val_loss: {comb_loss}, best_val_loss: {best_val_loss}, train_perp: {train_perplexity_}, valid_prep: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
                if args.etab_loss or args.etab_loss_only:
                    f.write(f'\ttrain_nlcpl: {train_nlcpl}, valid_nlcpl: {validation_nlcpl}\n')

            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train_loss: {train_comb_loss}, val_loss: {comb_loss}, best_val_loss: {best_val_loss}, train_perp: {train_perplexity_}, valid_prep: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            if args.etab_loss or args.etab_loss_only:
                print(f'\ttrain_nlcpl: {train_nlcpl}, valid_nlcpl: {validation_nlcpl}')

            if comb_loss < best_val_loss:
                checkpoint_filename_last = base_folder+'model_weights/epoch_best.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'noise_type': args.noise_type,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)
                best_val_loss = comb_loss
            
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
    argparser.add_argument("--path_for_meta_data", type=str, default="my_path/pdb_2021aug02_meta", help="path for loading meta data")
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
    argparser.add_argument("--augment_lim", type=float, default=1.0, help='RMSD limit for noise')
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--replicate", type=int, default=1, help='replicate to use in setting seed for noise')
    argparser.add_argument("--feat_type", type=str, default="protein_mpnn", help="type of featurizer to use")
    argparser.add_argument("--etab_loss", type=int, default=0, help="whether to use Potts model loss")
    argparser.add_argument("--etab_loss_only", type=int, default=0, help="whether to only use Potts model loss")
    argparser.add_argument("--output_dim", type=int, default=400, help="Potts model output dimension")
    argparser.add_argument("--node_self_sub", type=str, default=None, help="whether to use sequence probabilities as Potts model self energies")
    argparser.add_argument("--clone", type=int, default=1, help="whether to clone node tensors if using for Potts model self energies")
    argparser.add_argument("--seq_encoding", type=str, default="one_hot", help="Sequence encoding to use")
    argparser.add_argument("--num_workers", type=int, default=12, help="number of workers to use for data loading")
    argparser.add_argument("--struct_predict", type=int, default=0, help="whether to use end-to-end structure supervision")
    argparser.add_argument("--use_struct_weights", type=int, default=0, help="whether to use pre-trained weights for end-to-end structure supervision")
    argparser.add_argument("--multimer_structure_module", type=int, default=0, help="whether to use multimer model for end-to-end structure supervision")
    argparser.add_argument("--struct_predict_pairs", type=int, default=1, help="whether to use pairs for end-to-end structure supervision")
    argparser.add_argument("--struct_predict_seq", type=int, default=1, help="whether to use sequence for end-to-end structure supervision")
    argparser.add_argument("--load_optimizer", type=int, default=1, help="whether to load optimizer when fine-tuning a model")
    argparser.add_argument("--strict", type=int, default=1, help="enforce match between path for old model and current model")
    argparser.add_argument('--use_seq', type=int, default=1, help='whether to use sequence info in autoregressive decoding')
    argparser.add_argument("--consensus_seqs", type=str, default='', help="whether to use consensus sequences for sequence prediction")
    argparser.add_argument("--msa_seqs", type=int, default=0, help="whether to use msa sequences for sequence prediction")
    argparser.add_argument("--remove_missing", type=int, default=1, help="whether to remove residues missing structure information")
    args = argparser.parse_args()

    args.etab_loss = args.etab_loss == 1
    args.etab_loss_only = args.etab_loss_only == 1
    args.struct_predict = args.struct_predict == 1
    args.use_struct_weights = args.use_struct_weights == 1
    args.multimer_structure_module = args.multimer_structure_module == 1
    args.struct_predict_pairs = args.struct_predict_pairs == 1
    args.struct_predict_seq = args.struct_predict_seq == 1
    args.load_optimizer = args.load_optimizer == 1
    args.strict = args.strict == 1
    args.clone = args.clone == 1
    args.use_seq = args.use_seq == 1
    args.msa_seqs = args.msa_seqs == 1
    args.remove_missing = args.remove_missing == 1

    print('starting')    

    main(args)   
