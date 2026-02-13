import argparse
import os.path

def main(args):
    import torch
    from torch.utils.data import DataLoader
    import esm as esmlib   
    from utils import worker_init_fn, get_all_pdbs, loader_all_pdb, build_training_clusters, PDB_all_dataset, StructureDataset, StructureLoader, StructureSampler
    from model_utils import featurize, loss_smoothed, nlcpl, loss_nll, get_std_opt, ProteinMPNN
    import pickle
    import gzip

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : 3.5, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 2}

    print('loaded args')
    print(args)
    train, valid, test = build_training_clusters(params, debug=False)
    print('built clusters')

    train_list = list(train.keys())
    valid_list = list(valid.keys())
        
    # train_set = PDB_all_dataset(train_list, loader_all_pdb, train, params)
    # train_loader = DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    # valid_set = PDB_all_dataset(valid_list, loader_all_pdb, valid, params)
    # valid_loader = DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    if not os.path.exists(args.path_for_outputs):
        os.makedirs(args.path_for_outputs)

    print('loaded loaders')

    ## Load ESM if required
    if args.seq_encoding == 'esm2-150M':
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        esm_embed_layer = 30
        esm_embed_dim = 640
        one_hot = False
    elif args.seq_encoding == 'esm2-650M':
        esm, alphabet = esmlib.pretrained.esm2_t33_650M_UR50D()
        esm_embed_layer = 33
        esm_embed_dim = 1280
        one_hot = False
    elif args.seq_encoding == 'esm2-3B':
        esm, alphabet = esmlib.pretrained.esm2_t36_3B_UR50D()
        esm_embed_layer = 36
        esm_embed_dim = 2560
        one_hot = False
    elif args.seq_encoding == 'esm2-one_hot':
        esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
        esm_embed_layer = 0
        esm_embed_dim = 22
        one_hot = True
    elif args.seq_encoding == 'one_hot':
        esm, alphabet, batch_converter, esm_embed_layer, esm_embed_dim = None, None, None, None, None
        one_hot = True
    else:
        print("seq_encoding not recognized")
        raise ValueError
    if esm is not None:
        esm = esm.to(device=device)
        batch_converter = alphabet.get_batch_converter()
        esm = esm.eval()
    print('Loaded ESM: ', args.seq_encoding)

    print('about to iterate')    
    pdb, chains = get_all_pdbs(train_list, train, params, args.path_for_outputs, device, args.seq_encoding, esm, batch_converter, esm_embed_dim, esm_embed_layer, one_hot)
    print('cchains: ', pdb, chains)
    
    print('about to val iterate')
    pdb, chains = get_all_pdbs(valid_list, valid, params, args.path_for_outputs, device, args.seq_encoding, esm, batch_converter, esm_embed_dim, esm_embed_layer, one_hot)
    print('val cchains: ', pdb, chains)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--path_for_outputs", type=str, default="./exp_020", help="path for esm data")
    argparser.add_argument("--seq_encoding", type=str, default="one_hot", help="Sequence encoding to use")
     
    args = argparser.parse_args()
    print('starting')    

    main(args)   
