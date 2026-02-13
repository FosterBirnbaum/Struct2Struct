
import numpy as np
import argparse
import os.path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tqdm import tqdm
import pickle
import json, time, os, sys, glob
import shutil
import warnings
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
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor 
import esm as esmlib
from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, StructureSampler
from model_utils import featurize, loss_smoothed, nlcpl, structure_loss, loss_nll, potts_singlesite_loss, get_std_opt, ProteinMPNN

class AverageMeter:
    """Computes and stores the average and current value efficiently.
       Updates internal stats only every `update_every` calls.
    """
    def __init__(self, update_every=100):
        self.update_every = update_every
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self._iters = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._iters += 1

        if self._iters % self.update_every == 0:
            self.avg = self.sum / self.count if self.count > 0 else 0

def load_pickle_stream(path):
    """Generator to yield objects one-by-one from a pickle file."""
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def start_workers(train_loader, valid_loader, base_folder, args, n_pairs=2):
    executor = ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context('spawn'))

    train_futures = {}
    valid_futures = {}
    
    for i in range(n_pairs):
        f_train = executor.submit(get_pdbs, train_loader, i, 'train', base_folder, 1,
                                  args.max_protein_length, args.num_examples_per_epoch,
                                  args.consensus_seqs, args.msa_match_dict,
                                  args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                  args.single_species_sample, args.remove_missing,
                                  args.id_thresh, args.del_thresh, args.insrt_thresh)
        train_futures[f_train] = i

        f_valid = executor.submit(get_pdbs, valid_loader, i, 'valid', base_folder, 1,
                                  args.max_protein_length, args.num_examples_per_epoch,
                                  args.consensus_seqs, args.msa_match_dict,
                                  args.complex_mapping_path, args.msa_dir, args.msa_seqs,
                                  args.single_species_sample, args.remove_missing,
                                  args.id_thresh, args.del_thresh, args.insrt_thresh)
        valid_futures[f_valid] = i

    return executor, train_futures, valid_futures

def main(args):


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

    data_path = args.path_for_training_data
    data_meta_path = args.path_for_meta_data
    if (len(args.consensus_seqs) > 0) or ('ing' in args.msa_dir):
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
        args.batch_size = 1

    print('loaded args')
    print(args)
    if args.soluble_mpnn:
        exclude_df = pd.read_csv(args.soluble_mpnn)
        exclude = list(exclude_df['PDB_IDS'].values)
    else:
        exclude = []
    if args.exclude_msa:
        with open(args.exclude_msa, 'r') as f:
            exclude_msa = f.readlines()
        exclude += [cid.strip() for cid in exclude_msa]

    train, valid, test = build_training_clusters(params, args.debug, exclude)
    print('built clusters')

    if args.debug:
        train_list = list(train.keys())[:100]
        valid_list = list(valid.keys())[:100]
        print('debug len: ', len(train_list), len(valid_list))
    else:
        train_list = list(train.keys())
        valid_list = list(valid.keys())
        
    train_set = PDB_dataset(train_list, loader_pdb, train, params)
    train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(valid_list, loader_pdb, valid, params)
    valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    print('loaded loaders')

    if args.debug:
        for i, _ in enumerate(train_loader):
            pass
        for i_valid, _ in enumerate(valid_loader):
            pass
        print('debug loader len: ', i, i_valid)
    use_potts = args.etab_loss or args.etab_loss_only or args.etab_singlesite_loss
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    vocab = 21 if not args.msa_seqs else 22
    print('vocab: ', vocab)
    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise,
                        augment_type=args.noise_type,
                        augment_lim=args.augment_lim,
                        feat_type=args.feat_type,
                        use_potts=use_potts,
                        output_dim=args.output_dim,
                        node_self_sub=args.node_self_sub,
                        clone=args.clone,
                        seq_encoding=args.seq_encoding,
                        struct_predict=args.struct_predict,
                        use_struct_weights=args.use_struct_weights,
                        multimer_structure_module=args.multimer_structure_module,
                        struct_predict_pairs=args.struct_predict_pairs,
                        struct_predict_seq=args.struct_predict_seq,
                        use_seq=args.use_seq,
                        vocab=vocab,
                        num_letters=vocab,
                        device=device
                        )
    model.to(device)

    print('setup model')
    if PATH:
        checkpoint = torch.load(PATH, map_location=device, weights_only=False)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=args.strict)
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    if PATH and args.load_optimizer:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('setup optimizer')
    kwargs = {}
    kwargs['num_workers'] = args.num_workers
    kwargs['multiprocessing_context'] = 'spawn'

    executor, train_futures, valid_futures = start_workers(train_loader, valid_loader, base_folder, args, n_pairs=100)

               
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
    argparser.add_argument("--mixed_precision", type=int, default=1, help="train with mixed precision")
    argparser.add_argument("--replicate", type=int, default=1, help='replicate to use in setting seed for noise')
    argparser.add_argument("--feat_type", type=str, default="protein_mpnn", help="type of featurizer to use")
    argparser.add_argument("--etab_loss", type=int, default=0, help="whether to use Potts model loss")
    argparser.add_argument("--etab_singlesite_loss", type=int, default=0, help="whether to use singlesite Potts model loss")
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
    argparser.add_argument("--single_species_sample", type=int, default=0, help="whether to restrict MSA sampling to only 1 sequence per species")
    argparser.add_argument("--msa_dir", type=str, default='', help='path to MSAs')
    argparser.add_argument("--msa_match_dict", type=str, default='', help='mapping of chain ids for PDB MSAs')
    argparser.add_argument("--complex_mapping_path", type=str, default='', help='mapping of complex chain ids for PDB MSAs')
    argparser.add_argument("--msa_batch_size", type=int, default=1, help="batch size for msa sequences")
    argparser.add_argument("--exclude_msa", type=str, default='', help='PDBs to exclude because of missing MSA information')
    argparser.add_argument("--id_thresh", type=float, default=0.5, help="sequence identity cutoff for msa sequences")
    argparser.add_argument("--del_thresh", type=float, default=0.2, help="deletion percent cutoff for msa sequences")
    argparser.add_argument("--insrt_thresh", type=float, default=0.2, help="insertion percent cutoff for msa sequences")
    argparser.add_argument("--remove_missing", type=int, default=1, help="whether to remove residues missing structure information")
    argparser.add_argument("--soluble_mpnn", type=str, default='', help='path for pdb_ids to exclude when training a soluble version of the model')
    args = argparser.parse_args()

    args.etab_loss = args.etab_loss == 1
    args.etab_singlesite_loss = args.etab_singlesite_loss == 1
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
    args.single_species_sample = args.single_species_sample == 1
    args.remove_missing = args.remove_missing == 1
    args.mixed_precision = args.mixed_precision == 1

    print('starting')    

    main(args)   
