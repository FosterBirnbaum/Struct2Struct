import sys
import time
import torch
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence


# Extract kNN info
def extract_knn(X, mask, eps, top_k):
    # Convolutional network on NCHW
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    # Identify k nearest neighbors (including self)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)
    return mask_2D, D_neighbors, E_idx

def extract_edge_mapping(E_idx, mapping, max_num_edges):
    k = E_idx.shape[2]
    edge_idx = torch.zeros((mapping.shape[0], max_num_edges, 2*k), dtype=torch.int64)
    node_endpoint_idx = torch.zeros((mapping.shape[0], max_num_edges, 2*k), dtype=torch.int64)
    node_neighbors_idx = torch.zeros((mapping.shape[0], max_num_edges, 2*k), dtype=torch.int64)
    mapping_r = torch.reshape(mapping, E_idx.shape)
    for b in range(mapping_r.shape[0]):
        for i in range(mapping_r.shape[1]):
            for ij in range(mapping_r.shape[2]):
                j = E_idx[b,i,ij].item()
                mapping_i = mapping_r[b,i,ij]
                if i > j and i in E_idx[b,j]:
                    continue
                edge_idx[b, mapping_i, :k] = mapping_r[b,i]
                node_endpoint_idx[b, mapping_i, :k] = i*torch.ones(node_endpoint_idx[b, mapping_i,:k].shape)
                node_neighbors_idx[b, mapping_i, :k] = E_idx[b,i]
                edge_idx[b, mapping_i, k:] = mapping_r[b,j]
                node_endpoint_idx[b, mapping_i, k:] = j*torch.ones(node_endpoint_idx[b, mapping_i,:k].shape)
                node_neighbors_idx[b, mapping_i, k:] = E_idx[b,j]
    return edge_idx, node_endpoint_idx, node_neighbors_idx

def get_merge_dups_mask(E_idx):
    N = E_idx.shape[1]
    if E_idx.is_cuda:
        tens_place = torch.arange(N).cuda().unsqueeze(0).unsqueeze(-1)
    else:
        tens_place = torch.arange(N).unsqueeze(0).unsqueeze(-1)
    # tens_place = tens_place.unsqueeze(0).unsqueeze(-1)
    min_val = torch.minimum(E_idx, tens_place)
    max_val = torch.maximum(E_idx, tens_place)
    edge_indices = min_val*N + max_val
    edge_indices = edge_indices.flatten(1,2)
    unique_inv = []
    all_num_edges = []
    for b in range(len(edge_indices)):
        uniq, inv = torch.unique(edge_indices[b], return_inverse=True)
        unique_inv.append(inv)
        all_num_edges.append(len(uniq))
    unique_inv = torch.stack(unique_inv)
    return unique_inv, all_num_edges


def get_msa_paired_stats(msa, E_idx):
    pair_etab = torch.zeros((len(msa[0])), E_idx.shape[-1], 22, 22)
    for i_pos in range(len(msa[0])):
        for ii_pos, j_pos in enumerate(E_idx[i_pos]):
            dup = False
            if i_pos in E_idx[j_pos]:
                try:
                    jj_pos = (E_idx[j_pos] == i_pos).nonzero(as_tuple=True)[0].item()
                except Exception as e:
                    jj_pos = (E_idx[j_pos] == i_pos).nonzero(as_tuple=True)[0][0].item()
                dup = True
            if dup and j_pos > i_pos:
                continue
            cur_pos_etab = torch.zeros((22,22))
            for seq in msa:
                cur_pos_etab[seq[i_pos], seq[j_pos]] += 1
            cur_pos_etab = torch.div(cur_pos_etab, msa.shape[0])
            pair_etab[i_pos, ii_pos] = cur_pos_etab
            if dup:
                pair_etab[j_pos, jj_pos] = cur_pos_etab.transpose(-1,-2)
    return pair_etab


def gather_edges(edges, neighbor_idx):
    """ Gather the edge features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    edges : torch.Tensor
        The edge features in dense form
        Shape: n_batch x n_res x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    edge_features : torch.Tensor
        The gathered edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def _esm_featurize(chain_lens, seq, esm, batch_converter, use_esm_attns, esm_embed_layer, from_rla=False, use_reps=True, connect_chains=False, one_hot=False):
    esm_data = []
    scl = 0
    if one_hot:
        return str_to_1hot(seq).to(dtype=torch.float), None
    if not connect_chains:
        for ic, cl in enumerate(chain_lens):
            esm_data.append((str(ic), "".join(list(seq)[scl:scl+cl])))
            scl += cl
    else:
        esm_seq = []
        for ic, cl in enumerate(chain_lens):
            esm_seq.append("".join(list(seq)[scl:scl+cl]))
            scl += cl
        mask_seq = (25*"-").join(esm_seq)
        esm_seq = (25*"G").join(esm_seq)
        mask = torch.from_numpy(np.array(list(mask_seq)) != "-")
        esm_data.append(('prot', esm_seq))
    _, _, batch_tokens = batch_converter(esm_data)
    with torch.no_grad():
        if not from_rla:
            results = esm(batch_tokens, repr_layers=[esm_embed_layer], return_contacts=use_esm_attns)
        else:
            results = esm(batch_tokens, output_attentions=use_esm_attns)
    
    if use_reps:
        if not from_rla:
            emb_results = results['representations'][esm_embed_layer]
        else:
            emb_results = results['last_hidden_state']
    else:
        emb_results = results['logits']

    if not connect_chains:
        emb_results = [emb_result[1:chain_lens[ic] + 1] for ic, emb_result in enumerate(emb_results)]
        embs = torch.cat(emb_results)
    else:
        embs = emb_results[0][1:-1]
        embs = embs[mask]
    if not use_esm_attns:
        return embs, None
    if not connect_chains:
        scl = 0
        if from_rla:
            attns = torch.cat(results['attentions'], 1)
            attns = [attn[:,1:chain_lens[ic] + 1, 1:chain_lens[ic] + 1] for ic, attn in enumerate(attns)]
            tot_len = sum([attns[i].size(1) for i in range(len(attns))])
            final_attns = torch.zeros(tot_len, tot_len, attns[0].size(0))
            for ic, attn in enumerate(attns):
                final_attns[scl:scl+chain_lens[ic], scl:scl+chain_lens[ic], :] = attn.permute(1,2,0)
                scl += chain_lens[ic]
        else:
            sha = results['attentions'].shape
            attns = results['attentions'].contiguous().view(sha[0], -1, sha[3], sha[4])
            # assert(attns.shape[0] == 1)
            attns = [attn[:,1:chain_lens[ic] + 1, 1:chain_lens[ic] + 1] for ic, attn in enumerate(attns)]
            tot_len = sum([attns[i].size(1) for i in range(len(attns))])
            final_attns = torch.zeros(tot_len, tot_len, attns[0].size(0))
            for ic, attn in enumerate(attns):
                final_attns[scl:scl+chain_lens[ic], scl:scl+chain_lens[ic], :] = attn.permute(1,2,0)
                scl += chain_lens[ic]
    else:
        if not from_rla:
            sha = results['attentions'].shape
            attns = results['attentions'].contiguous().view(sha[0], -1, sha[3], sha[4])[0,:,1:-1,1:-1]
            attns = attns.permute(1,2,0)
            attns = attns[mask]
            final_attns = attns[:,mask,:]
        else:
            attns = torch.cat(results['attentions'], 1)[0,:,1:-1,1:-1].permute(1,2,0)
            attns = attns[mask]
            final_attns = attns[:,mask,:]
    return embs, final_attns


# zero is used as padding
AA_to_int = {
    'A': 1,
    'ALA': 1,
    'C': 2,
    'CYS': 2,
    'D': 3,
    'ASP': 3,
    'E': 4,
    'GLU': 4,
    'F': 5,
    'PHE': 5,
    'G': 6,
    'GLY': 6,
    'H': 7,
    'HIS': 7,
    'I': 8,
    'ILE': 8,
    'K': 9,
    'LYS': 9,
    'L': 10,
    'LEU': 10,
    'M': 11,
    'MET': 11,
    'N': 12,
    'ASN': 12,
    'P': 13,
    'PRO': 13,
    'Q': 14,
    'GLN': 14,
    'R': 15,
    'ARG': 15,
    'S': 16,
    'SER': 16,
    'T': 17,
    'THR': 17,
    'V': 18,
    'VAL': 18,
    'W': 19,
    'TRP': 19,
    'Y': 20,
    'TYR': 20,
    '-': 21,
    'X': 22
}

esm_list = [ 5, 23, 13,  9, 18,  6, 21, 12, 15,  4, 20, 17, 14, 16, 10,  8, 11,
          7, 22, 19, 30, 24] # Alphabetical
esm_encodings = {}
for i, e in enumerate(esm_list):
    esm_encodings[i] = e
esm_decodings = {}
for i, e in enumerate(esm_list):
    esm_decodings[e] = i

AA_to_int = {key: val - 1 for key, val in AA_to_int.items()}

int_to_AA = {y: x for x, y in AA_to_int.items() if len(x) == 1}

int_to_3lt_AA = {y: x for x, y in AA_to_int.items() if len(x) == 3}

def seq_to_ints(sequence):
    """
    Given a string of one-letter encoded AAs, return its corresponding integer encoding
    """
    return [AA_to_int[residue] for residue in sequence]


def ints_to_seq(int_list):
    return [int_to_AA[i] for i in int_list]

def aa_three_to_one(residue):
    return int_to_AA[AA_to_int[residue]]

def esm_convert(seq):
    return [esm_encodings[s] for s in seq]

def esm_deconvert(seq):
    return [esm_decodings[s] for s in seq]

def ints_to_seq_torch(seq):
    return "".join(ints_to_seq(seq.cpu().numpy()))

def esm_ints_to_seq_torch(seq):
    return "".join(ints_to_seq(esm_deconvert(seq)))

def ints_to_seq_normal(seq):
    return "".join(ints_to_seq(seq))

DEFAULT_MODEL_HPARAMS = {
    'model': 'multichain',
    'matches': 'transformer',
    'term_hidden_dim': 32,
    'flex_hidden_dim': 1,
    'condense_options': '',
    'energies_hidden_dim': 32,
    'gradient_checkpointing': True,
    'cov_features': 'all_raw',
    'cov_compress': 'ffn',  #
    'num_pair_stats': 28,  #
    'num_sing_stats': 0,  #
    'resnet_blocks': 4,  #
    'term_layers': 4,  #
    'flex_layers': 4, #
    'term_heads': 4,  #
    'conv_filter': 3,  #
    'matches_layers': 4,  #
    'matches_num_heads': 4,  #
    'k_neighbors': 30,  #
    'k_cutoff': None,  #
    'contact_idx': True,  #
    'cie_dropout': 0.1,  #
    'cie_scaling': 500,  #
    'cie_offset': 0,  #
    'transformer_dropout': 0.1,  #
    'term_use_mpnn': True,  #
    'energies_protein_features': 'full',  #
    'energies_augment_eps': 0,  #
    'energies_encoder_layers': 6,  #
    'energies_dropout': 0.1,  #
    'energies_use_mpnn': False,  #
    'energies_output_dim': 20 * 20,  #
    'energies_gvp': False,  #
    'energies_graphformer': False, #
    'energies_geometric': False,  #
    'energies_full_graph': True,  #
    'res_embed_linear': False,  #
    'matches_linear': False,  #
    'term_mpnn_linear': False,  #
    'struct2seq_linear': False,
    'use_terms': True,  #
    'use_flex': False,
    'use_coords': True,
    'num_features':
    len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len']),  #
    'chain_handle': '',
    'test_code': '',
    'edge_update': '',
    'edge_update_k': 0,
    'energies_graph_type': 'undirected',
    'loss_graph_type': 'undirected',
    'activation_layers': 'relu',
    'voxel_max': 15,
    'voxel_size': 16,
    'voxel_width': 2,
    'interaction_mlp_layers': 3,
    'interaction_mlp_in': [],
    'interaction_mlp_out': [],
    'mlp_dropout': 0,
    'skip_attention': False,
    'kernel_width': 0,
    'kernel_sigma': 1,
    'masking': '',
    'neighborhood_multiplier': 2,
    'graphformer_num_heads': 4,
    'graphformer_num_layers': 3,
    'graphformer_mlp_multiplier': 1,
    'graphformer_edge_type': 'basic',
    'graphformer_dropout': 0,
    'num_positional_embeddings': 16,
    'edge_merge_fn': 'default',
    'energy_merge_fn': 'default',
    'featurizer': 'multichain',
    'side_chain': False,
    'nonlinear_features': False,
    'esm_feats': False,
    'rla_feats': False,
    'esm_embed_layer': 30,
    'esm_embed_dim':  640,
    'esm_rep_feat_ins': [640],
    'esm_rep_feat_outs': [32],
    'esm_attn_feat_ins': [600, 100],
    'esm_attn_feat_outs': [100, 20],
    'esm_model': 150,
    'convert_to_esm': False,
    'old': False,
    'one_hot': False,
    'nodes_to_probs': False,
    'edges_to_seq': False
}

DEFAULT_TRAIN_HPARAMS = {
    'term_matches_cutoff': None,
    # 'test_term_matches_cutoff': None,
    # ^ is an optional hparam if you want to use a different TERM matches cutoff during validation/testing vs training
    'train_batch_size': 16,
    'shuffle': True,
    'sort_data': True,
    'semi_shuffle': False,
    'regularization': 0,
    'max_term_res': 55000,
    'max_seq_tokens': None,
    'min_seq_tokens': 30,
    'term_dropout': None,
    'loss_config': {
        'nlcpl': 1
    },
    'finetune': False,
    'lr_multiplier': 1,
    'finetune_lr': 1e-6,
    'bond_length_noise_level': 0,
    'bond_angle_noise_level': 0,
    'patience': 20,
    'pdb_dataset': None,
    'flex_folder': None,
    'num_ensembles': 1,
    'msa_type': '',
    'msa_id_cutoff': 0.5,
    'msa_depth_lim': 60,
    'undirected_edge_scale': 1,
    'flex_type': '',
    'noise_level': 0.0,
    'replicate': 1,
    'noise_lim': 2,
    'use_sc': False,
    'sc_mask_rate': 0.15,
    'base_sc_mask': 0.05,
    'sc_mask': [],
    'chain_mask': False,
    'sc_mask_schedule': False,
    'sc_info': 'full',
    'sc_noise': 0,
    'mask_neighbors': False,
    'mask_interface': False,
    'half_interface': True,
    'inter_cutoff': 8,
    'warmup': 4000,
    'post_esm_mask': False,
    'use_esm_attns': False,
    'use_reps': False,
    'connect_chains': True,
    'from_wds': True,
    'num_recycles': 0,
    'keep_seq_recycle': True,
    'sc_screen': False,
    'reload_data_every_n_epochs': 2
}

def backwards_compat(model_hparams, run_hparams):
    if "cov_features" not in model_hparams.keys():
        model_hparams["cov_features"] = False
    if "term_use_mpnn" not in model_hparams.keys():
        model_hparams["term_use_mpnn"] = False
    if "matches" not in model_hparams.keys():
        model_hparams["matches"] = "resnet"
    if "struct2seq_linear" not in model_hparams.keys():
        model_hparams['struct2seq_linear'] = False
    if "energies_gvp" not in model_hparams.keys():
        model_hparams['energies_gvp'] = False
    if "num_sing_stats" not in model_hparams.keys():
        model_hparams['num_sing_stats'] = 0
    if "num_pair_stats" not in model_hparams.keys():
        model_hparams['num_pair_stats'] = 0
    if "contact_idx" not in model_hparams.keys():
        model_hparams['contact_idx'] = False
    if "fe_dropout" not in model_hparams.keys():
        model_hparams['fe_dropout'] = 0.1
    if "fe_max_len" not in model_hparams.keys():
        model_hparams['fe_max_len'] = 1000
    if "cie_dropout" not in model_hparams.keys():
        model_hparams['cie_dropout'] = 0.1
    if "energy_merge_fn" not in model_hparams.keys():
        model_hparams['energy_merge_fn'] = 'default'
    if "edge_merge_fn" not in model_hparams.keys():
        model_hparams["edge_merge_fn"] = 'default'
    if "featurizer" not in model_hparams.keys():
        model_hparams["featurizer"] = 'multichain'
    if "nonlinear_features" not in model_hparams.keys():
        model_hparams["nonlinear_features"] = False
    if "esm_feats" not in model_hparams.keys():
        model_hparams["esm_feats"] = False
    if "rla_feats" not in model_hparams.keys():
        model_hparams["rla_feats"] = False
    if "esm_embed_layer" not in model_hparams.keys():
        model_hparams["esm_embed_layer"] = 30
    if "esm_embed_dim" not in model_hparams.keys():
        model_hparams["esm_embed_dim"] = 640
    if "esm_rep_feat_ins" not in model_hparams.keys():
        model_hparams["esm_rep_feat_ins"] = [640]
    if "esm_rep_feat_outs" not in model_hparams.keys():
        model_hparams["esm_rep_feat_outs"] = [32]
    if "esm_attn_feat_ins" not in model_hparams.keys():
        model_hparams["esm_attn_feat_ins"] = [600, 100]
    if "esm_attn_feat_outs" not in model_hparams.keys():
        model_hparams["esm_attn_feat_outs"] = [100, 20]
    if "esm_model" not in model_hparams.keys():
        model_hparams["esm_model"] = '150'
    if "sc_info" not in run_hparams.keys():
        run_hparams["sc_info"] = 'full'
    if "use_esm_attns" not in run_hparams.keys():
        run_hparams["use_esm_attns"] = False
    if "use_reps" not in run_hparams.keys():
        run_hparams["use_reps"] = False
    if "connect_chains" not in run_hparams.keys():
        run_hparams["connect_chains"] = False
    if "use_sc" not in run_hparams.keys():
        run_hparams["use_sc"] = False
    if "old" not in model_hparams.keys():
        model_hparams['old'] = True
    if "mask_interface" not in run_hparams.keys():
        run_hparams["mask_interface"] = False
    if "num_recycles" not in run_hparams.keys():
        run_hparams["num_recycles"] = 0
    if "use_edges_for_nodes" not in model_hparams.keys():
        model_hparams["use_edges_for_nodes"] = False
    if "mask_neighbors" not in run_hparams.keys():
        run_hparams["mask_neighbors"] = False
    if "convert_to_esm" not in model_hparams.keys():
        model_hparams["convert_to_esm"] = False
    if "one_hot" not in model_hparams.keys():
        model_hparams["one_hot"] = False
    if "keep_seq_recycle" not in run_hparams.keys():
        run_hparams["keep_seq_recycle"] = True
    if "nodes_to_probs" not in model_hparams.keys():
        model_hparams["nodes_to_probs"] = True
    if "sc_screen" not in run_hparams.keys():
        run_hparams["sc_screen"] = False
    if "half_interface" not in run_hparams.keys():
        run_hparams["half_interface"] = True
    if "inter_cutoff" not in run_hparams.keys():
        run_hparams["inter_cutoff"] = 8
    if "msa_id_cutoff" not in run_hparams.keys():
        run_hparams["msa_id_cutoff"] = 0.5
    if "edges_to_seq" not in model_hparams.keys():
        model_hparams["edges_to_seq"] = False
    if "post_esm_mask" not in run_hparams.keys():
        run_hparams["post_esm_mask"] = False
    if "sc_mask_rate" not in run_hparams.keys():
        run_hparams['sc_mask_rate'] = 0.0
    if "sc_mask_schedule" not in run_hparams.keys():
        run_hparams["sc_mask_schedule"] = False
    return model_hparams, run_hparams