from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint
from torch.nn.utils.rnn import pad_sequence
import scipy
from sklearn.neighbors import NearestNeighbors
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import re
from graph_feat import MultiChainProteinFeatures as CoordFeatures
from etab_utils import merge_duplicate_pairE, expand_etab

def char_to_1hot(char):
    ALPHABET='ACDEFGHIKLMNPQRSTVWY-X'
    char_idx = ALPHABET.find(char)
    if char_idx == -1:
        return None  # Character not found in the alphabet
    one_hot = np.zeros(len(ALPHABET))
    one_hot[char_idx] = 1
    return one_hot

def str_to_1hot(input_string):
    embedding = []
    for char in input_string:
        one_hot = char_to_1hot(char)
        if one_hot is not None:
            embedding.append(torch.from_numpy(one_hot))
    return torch.stack(embedding)

def str_to_list(s):
    pattern = r'<[^>]+>|.'
    
    return re.findall(pattern, s)

def _esm_featurize(seq, chain_lens, esm, batch_converter, esm_embed_layer, device, one_hot=False, linker_length=25):
    with torch.no_grad():
        esm_data = []
        scl = 0
        if one_hot:
            return str_to_1hot("".join(seq)).to(dtype=torch.float)
        esm_seq = []
        for ic, cl in enumerate(chain_lens):
            esm_seq.append("".join(seq[scl:scl+cl]))
            scl += cl
        mask_seq = (linker_length*"/").join(esm_seq)
        esm_seq = (linker_length*"G").join(esm_seq)
        mask = torch.from_numpy(np.array(str_to_list(mask_seq)) != "/")
        esm_data.append(('prot', esm_seq))
        _, _, batch_tokens = batch_converter(esm_data)
        batch_tokens = batch_tokens.to(device=device)
        results = esm(batch_tokens, repr_layers=[esm_embed_layer], return_contacts=False)
        emb_results = results['representations'][esm_embed_layer]
            
        embs = emb_results[0][1:-1]
        embs = embs[mask]
    return embs



def _extract_npz_array(npz_obj, keys, fallback_index=None):
    for k in keys:
        if k in npz_obj.files:
            return npz_obj[k]
    if fallback_index is not None and len(npz_obj.files) > fallback_index:
        return npz_obj[npz_obj.files[fallback_index]]
    return None


def _load_esmc_npz_for_protein(embeddings_dir, name):
    if not embeddings_dir:
        return None
    npz_path = os.path.join(embeddings_dir, f'embeddings_{name}.npz')
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    msa = _extract_npz_array(data, ['msa_embeddings', 'msa', 'arr_0'], fallback_index=0)
    if msa is None:
        return None
    if msa.ndim == 3:
        msa = msa.mean(axis=1)
    return {'msa': msa.astype(np.float32)}


def _load_esmc_npz_for_length(embeddings_dir, target_length, length_npz_cache, available_lengths):
    if not embeddings_dir:
        return None

    if target_length in length_npz_cache:
        return length_npz_cache[target_length]

    if target_length in available_lengths:
        chosen_length = target_length
    elif len(available_lengths) > 0:
        chosen_length = min(available_lengths, key=lambda x: abs(x - int(target_length)))
    else:
        length_npz_cache[target_length] = None
        return None

    npz_path = os.path.join(embeddings_dir, f'embeddings_len_{int(chosen_length)}.npz')
    if not os.path.exists(npz_path):
        length_npz_cache[target_length] = None
        return None

    data = np.load(npz_path)
    msa = _extract_npz_array(data, ['msa_embeddings', 'msa', 'arr_0'], fallback_index=0)
    if msa is None:
        length_npz_cache[target_length] = None
        return None
    if msa.ndim == 3:
        msa = msa.mean(axis=1)
    out = msa.astype(np.float32)
    length_npz_cache[target_length] = out
    return out


def _pick_real_negative_name(name, protein_to_negatives, available_names=None):
    candidates = protein_to_negatives.get(name, [])
    if available_names is not None:
        candidates = [prot_name for prot_name in candidates if prot_name in available_names]
    if candidates:
        return random.choice(candidates)
    return None


def build_esmc_batch_lookup(names, lengths, epoch, esmc_cache, protein_to_negatives, num_real_negatives_max=16, real_neg_warmup_epochs=50, esmc_embeddings_dir='', protein_clusters=None):
    if esmc_cache is None and not esmc_embeddings_dir:
        return None

    out = {}
    real_frac = min(float(epoch + 1) / float(max(real_neg_warmup_epochs, 1)), 1.0)
    n_real = int(round(real_frac * num_real_negatives_max))
    available_names = set(esmc_cache.keys()) if esmc_cache is not None else None
    available_lengths = set()
    if esmc_embeddings_dir:
        for npz_path in glob.glob(os.path.join(esmc_embeddings_dir, 'embeddings_len_*.npz')):
            m = re.search(r'embeddings_len_(\d+)\.npz$', os.path.basename(npz_path))
            if m:
                available_lengths.add(int(m.group(1)))
    length_npz_cache = {}

    for name, length in zip(names, lengths):
        info = esmc_cache.get(name) if esmc_cache is not None else None
        if info is None:
            info = _load_esmc_npz_for_protein(esmc_embeddings_dir, name)
        if info is None:
            continue

        random_rows = info.get('random')
        if random_rows is None:
            random_rows = _load_esmc_npz_for_length(esmc_embeddings_dir, int(length), length_npz_cache, available_lengths)

        item = {'msa': info.get('msa'), 'random': random_rows}
        real_list = []
        for _ in range(n_real):
            neg_name = _pick_real_negative_name(
                name,
                protein_to_negatives,
                available_names=available_names,
            )
            if neg_name is None:
                continue
            neg_info = esmc_cache.get(neg_name) if esmc_cache is not None else None
            if neg_info is None:
                neg_info = _load_esmc_npz_for_protein(esmc_embeddings_dir, neg_name)
            if neg_info is None:
                continue
            neg_rows = neg_info.get('msa')
            if neg_rows is not None and len(neg_rows) > 0:
                pick = neg_rows[np.random.randint(0, len(neg_rows))]
                real_list.append(pick)
        if len(real_list) > 0:
            item['real'] = np.stack(real_list, axis=0)
        out[name] = item
    return out


def load_pairformer_batch_lookup(names, lengths, pairformer_embeddings_dir, emb_dim=128):
    if not pairformer_embeddings_dir:
        return None, None

    pairformer_lookup = {}
    missing = []
    bad_shape = []
    for name, length in zip(names, lengths):
        npz_path = os.path.join(pairformer_embeddings_dir, f'embeddings_{name}.npz')
        if not os.path.exists(npz_path):
            missing.append(name)
            continue
        try:
            obj = np.load(npz_path)
            if 'z' not in obj.files:
                bad_shape.append(name)
                continue
            z = obj['z']
            if z.ndim != 3 or z.shape[0] != int(length) or z.shape[1] != int(length) or z.shape[2] != emb_dim:
                bad_shape.append(name)
                continue
            pairformer_lookup[name] = z.astype(np.float32)
        except Exception:
            bad_shape.append(name)

    if len(missing) > 0:
        print(f"| pairformer missing embeddings for {len(missing)} proteins in batch (example: {missing[0]})")
    if len(bad_shape) > 0:
        print(f"| pairformer malformed embeddings for {len(bad_shape)} proteins in batch (example: {bad_shape[0]})")
    return pairformer_lookup, emb_dim


def featurize(batch, device, augment_type, augment_eps, replicate, epoch, esm=None, batch_converter=None, esm_embed_dim=2560, esm_embed_layer=36, one_hot=False, return_esm=False, openfold_backbone=False, msa_seqs=False, msa_batch_size=10, esmc_cache=None, esmc_embeddings_dir='', esmc_protein_negatives=None, esmc_num_real_negatives_max=16, esmc_real_neg_warmup_epochs=50, esmc_protein_clusters=None, pairformer_embeddings_dir=""):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX-'

    if msa_seqs:
        new_batch = []
        masked_chains = batch[0]['masked_list']
        visible_chains = batch[0]['visible_list']
        all_chains = masked_chains + visible_chains
        all_seq = ''
        random.seed(epoch + np.sum([ord(char) for char in batch[0]['name']]))
        for chain_letter in all_chains:
            all_seq += batch[0][f'seq_chain_{chain_letter}'][0]
            random.shuffle(batch[0][f'seq_chain_{chain_letter}'])
        num_seqs = max(1, math.floor(msa_batch_size / len(all_seq)))
        
        for i_seq in range(num_seqs):
            b_copy = copy.deepcopy(batch[0])
            for chain_letter in all_chains:
            # chain_letter = all_chains[0]
                b_copy[f'seq_chain_{chain_letter}'] = batch[0][f'seq_chain_{chain_letter}'][i_seq % len(batch[0][f'seq_chain_{chain_letter}'])]
                # for seq in batch[0][f'seq_chain_{chain_letter}'][:num_seqs]:
                    
                    # b_copy[f'seq_chain_{chain_letter}'] = seq
                    # if (len(new_batch) + 1) * len(seq) >= msa_batch_size:
                    #     break
            new_batch.append(b_copy)
        batch = new_batch

    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/protein_mpnn_potts_ing_msa_50_20_batch_species_20_R1/log3.txt', 'a') as f:
    #     f.write(str(mask_self.shape) + '\n')
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    if esm is not None:
        S = np.zeros([B, L_max, esm_embed_dim], dtype=np.int32) #sequence AAs integers
    else:
        S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    S_true = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    all_chain_lens = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    names = []        
    if openfold_backbone:
        from openfold.utils import geometry, all_atom_multimer
        openfold_backbones = []
        
    for i, b in enumerate(batch):
        names.append(b['name'])
        chain_lens = []
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']

            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/protein_mpnn_potts_ing_msa_50_20_batch_species_20_R1/log2.txt', 'a') as f:
                #     f.write(str(l0) + ' ' + str(l1) + '\n')
                #     f.write(str(chain_length) + '\n')
                #     f.write(str(l1 - l0) + '\n')
                #     f.write(str(mask_self.shape) + '\n')
                #     f.write(b['name'] + '\n')
                # try:
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                # except Exception as e:
                #     with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/protein_mpnn_potts_ing_msa_50_20_batch_species_20_R1/log_err.txt', 'a') as f:
                #         f.write(str(l0) + ' ' + str(l1) + '\n')
                #         f.write(str(chain_length) + '\n')
                #         f.write(str(l1 - l0) + '\n')
                #         f.write(b['name'] + '\n')
                #     raise e
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            chain_lens.append(chain_length)
        all_chain_lens.append(chain_lens)
        try:
            x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        except:
            with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/protein_mpnn_potts_ing_consensus_R1/log_out2.txt', 'w') as f:
                f.write(b['name'] + '\n')
                f.write(all_chains)
                f.write(visible_chains)
                f.write(str(chain_coords))
            raise ValueError

        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        if openfold_backbone:
            isnan = np.isnan(x)
            x_0 = copy.deepcopy(x)
            x_0[isnan] = 0.
            atom_pos = geometry.Vec3Array.from_array(torch.from_numpy(x_0).to(torch.float32))
            rigid, _ = all_atom_multimer.make_backbone_affine(atom_pos, torch.ones(atom_pos.shape[0]), 7*torch.ones(atom_pos.shape[0], dtype=torch.long))
            openfold_backbones.append(rigid.to_tensor_4x4())

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        # X[i,:,:,:] = x_pad
        try:
            X[i,:,:,:] = x_pad
        except Exception as e:
            with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/protein_mpnn_potts_pdb_msa_50_20_20_batch_20_R3/log_mutils.txt', 'a') as f:
                f.write(b['name'] + '\n')
                f.write(all_sequence + '\n')
                f.write(str(len(all_sequence)) + '\n')
                f.write(str(X.shape) + '\n')
                f.write(str(x_pad.shape) + '\n')
                f.write(str(x.shape) + '\n')
                f.write(str(L_max) + ' ' + str(l) + ' ' + str(L_max - l) + '\n')
            raise e
        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad
        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        # try:
        #     indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        # except Exception as e:
        #     print(e)
        #     print(all_sequence)
        #     print(b['name'])
        #     return
        if esm is not None:
            esm_emb = _esm_featurize(all_sequence, chain_lens, esm, batch_converter, esm_embed_layer, device, one_hot=one_hot)
            if return_esm:
                return esm_emb, chain_lens
            S[i,:l] = esm_emb.cpu().numpy()
        else:
            S[i, :l] = indices
        S_true[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    if msa_seqs:
        mask = mask * (S != 21).astype(np.float32)
    X[isnan] = 0.

    # Add noise if needed
    # if augment_type == 'atomic':
    #     X = X + augment_eps * np.random.standard_normal(size=np.shape(X))
    # elif augment_type.find('torsion') > -1:
    #     for i in range(X.shape[0]):
    #         X[i] += generate_noise(augment_type, augment_eps, replicate, epoch, X[i], chain_lens=all_chain_lens[i], mask=mask[i])

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long)
    if esm is not None:
        S = torch.from_numpy(S).to(dtype=torch.float32)
    else:
        S = torch.from_numpy(S).to(dtype=torch.long)
    S_true = torch.from_numpy(S_true).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long)
    if openfold_backbone:
        backbone_4x4 = pad_sequence(openfold_backbones, batch_first=True, padding_value=0)
    else:
        backbone_4x4 = torch.Tensor([])

    esmc_batch_lookup = build_esmc_batch_lookup(names, lengths, epoch, esmc_cache, esmc_protein_negatives or {}, num_real_negatives_max=esmc_num_real_negatives_max, real_neg_warmup_epochs=esmc_real_neg_warmup_epochs, esmc_embeddings_dir=esmc_embeddings_dir, protein_clusters=esmc_protein_clusters)

    pairformer_lookup, pairformer_dim = load_pairformer_batch_lookup(names, lengths, pairformer_embeddings_dir, emb_dim=128)
    pairformer_z = np.zeros((B, L_max, L_max, pairformer_dim if pairformer_dim is not None else 128), dtype=np.float32)
    pairformer_mask = np.zeros((B, L_max, L_max), dtype=np.float32)
    if pairformer_lookup is not None:
        for i, name in enumerate(names):
            z = pairformer_lookup.get(name)
            if z is None:
                continue
            l = z.shape[0]
            pairformer_z[i, :l, :l, :] = z
            pairformer_mask[i, :l, :l] = 1.0

    pairformer_z = torch.from_numpy(pairformer_z).to(dtype=torch.float32)
    pairformer_mask = torch.from_numpy(pairformer_mask).to(dtype=torch.float32)

    return (X, S, S_true, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all, all_chain_lens, backbone_4x4, names, esmc_batch_lookup, pairformer_z, pairformer_mask)


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, vocab, weight=0.1, fixed_denom=2000.0):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, vocab).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    if fixed_denom != 0:
        loss_av = torch.sum(loss * mask) / fixed_denom #fixed 
    else:
        loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def nlcpl(etab, E_idx, S, mask, fixed_denom=0.0):
    """ Negative log composite psuedo-likelihood
        Averaged nlcpl per residue, across batches
        p(a_i,m ; a_j,n) =
            softmax [
                E_s(a_i,m) + E_s(a_j,n)
                + E_p(a_i,m ; a_j,n)
                + sum_(u != m,n) [
                    E_p(a_i,m; A_u)
                    + E_p(A_u, a_j,n)
                    ]
                ]

        Returns: log likelihoods per residue, as well as tensor mask
    """
    ref_seqs = S
    x_mask = mask

    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)

    isnt_x_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1] # b x L x 1 x 22 x 22
    pair_etab = etab[:, :, 1:] # b x L x 29 x 22 x 22

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1) # b x L x 1 x 22
    self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, k - 1, -1) # b x L x 29 x 22

    # E_idx for all but self
    E_idx_jn = E_idx[:, :, 1:] # b x L x 29

    # self Es gathered from E_idx_others
    E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, 22) # b x L x 29 x 22
    self_nrgs_jn = torch.gather(self_nrgs_im_expand, 1, E_idx_jn_expand) # b x L x 29 x 22

    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx_jn) # b x L x 29
    # expand the matrix so we can gather pair energies
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1, 1]).expand(-1, -1, -1, 22, -1) # b x L x 29 x 22 x 1
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    # sum_(u != n,m) E_p(a_i,n; A_u)
    sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2) # b x L x 22
    pair_nrgs_im_u = sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, k - 1, -1) - pair_nrgs_jn # b x L x 29 x 22

    # get pair_nrgs_u_jn from pair_nrgs_im_u
    E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand(pair_nrgs_im_u.shape) # b x L x 29 x 22
    pair_nrgs_u_jn = torch.gather(pair_nrgs_im_u, 1, E_idx_imu_to_ujn) # b x L x 29 x 22

    # start building this wacky energy table
    self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(-1, -1, -1, -1, 22) # b x L x 29 x 22 x 22
    self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 22).transpose(-2, -1) # b x L x 29 x 22 x 22
    pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(-1, -1, -1, -1, 22) # b x L x 29 x 22 x 22
    pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 22).transpose(-2, -1) # b x L x 29 x 22 x 22

    composite_nrgs = (self_nrgs_im_expand + self_nrgs_jn_expand + pair_etab + pair_nrgs_im_expand +
                      pair_nrgs_jn_expand) # b x L x 29 x 21 x 21

    # convert energies to probabilities
    composite_nrgs_reshape = composite_nrgs.view(n_batch, L, k - 1, 22 * 22, 1) # b x L x 29 x 484 x 1
    log_composite_prob_dist = torch.log_softmax(-composite_nrgs_reshape, dim=-2).view(n_batch, L, k - 1, 22, 22) # b x L x 29 x 22 x 22
    # get the probability of the sequence
    im_probs = torch.gather(log_composite_prob_dist, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1, 1]).expand(-1, -1, k - 1, 1) # b x L x 29 x 1
    log_edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1) # b x L x 29

    # reshape masks
    x_mask = x_mask.unsqueeze(-1) # b x L x 1
    isnt_x_aa = isnt_x_aa.unsqueeze(-1) # b x L x 1
    full_mask = x_mask * isnt_x_aa
    
    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    
    n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))
    if fixed_denom != 0:
        nlcpl_return = -1*torch.sum(log_edge_probs) / fixed_denom
    else:
        nlcpl_return = -1*torch.sum(log_edge_probs) / n_edges

    if nlcpl_return.isnan().item():
        if int(n_edges) == 0:
            return 0, 0
        print("ALARM")
        print(nlcpl_return)
        for i in range(full_mask.shape[0]):
            print('\t', torch.sum(full_mask[i]).item())
            print('\t', torch.sum((mask[i]) * (isnt_x_aa[i,:,0])).item())
        print(n_edges)
        print(etab.shape, etab.isnan().any())
        print(S)
        nlcpl_return = 0
        return nlcpl_return, -1

    return nlcpl_return, int(n_edges)


def gauge_fix_etab(etab, center_singlesite=True, center_pairwise=True):
    """Apply simple gauge-fixing transforms to Potts energy tables.

    Args:
        etab: [B, L, K, 400] tensor containing flattened 20x20 tables.
        center_singlesite: zero-center self-site diagonal energies.
        center_pairwise: apply zero-sum gauge to pairwise 20x20 matrices.
    """
    if not center_singlesite and not center_pairwise:
        return etab

    n_batch, L, k, n_out = etab.shape
    if n_out != 400:
        return etab

    etab_2d = etab.view(n_batch, L, k, 20, 20).clone()

    if center_singlesite:
        self_etab = etab_2d[:, :, 0]
        self_diag = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
        self_diag = self_diag - self_diag.mean(dim=-1, keepdim=True)
        self_etab.diagonal(offset=0, dim1=-2, dim2=-1).copy_(self_diag)

    if center_pairwise and k > 1:
        pair_etab = etab_2d[:, :, 1:]
        row_mean = pair_etab.mean(dim=-1, keepdim=True)
        col_mean = pair_etab.mean(dim=-2, keepdim=True)
        total_mean = pair_etab.mean(dim=(-2, -1), keepdim=True)
        etab_2d[:, :, 1:] = pair_etab - row_mean - col_mean + total_mean

    return etab_2d.reshape(n_batch, L, k, n_out)


def etab_l2_regularization(etab, singlesite_weight=0.0, pairwise_weight=0.0):
    """Return L2 regularization penalty for Potts energies."""
    if singlesite_weight <= 0.0 and pairwise_weight <= 0.0:
        return torch.zeros((), device=etab.device, dtype=etab.dtype)

    n_batch, L, k, n_out = etab.shape
    if n_out != 400:
        return torch.zeros((), device=etab.device, dtype=etab.dtype)

    etab_2d = etab.view(n_batch, L, k, 20, 20)
    penalty = torch.zeros((), device=etab.device, dtype=etab.dtype)

    if singlesite_weight > 0.0:
        self_diag = torch.diagonal(etab_2d[:, :, 0], offset=0, dim1=-2, dim2=-1)
        penalty = penalty + singlesite_weight * torch.mean(self_diag.square())

    if pairwise_weight > 0.0 and k > 1:
        penalty = penalty + pairwise_weight * torch.mean(etab_2d[:, :, 1:].square())

    return penalty

def potts_singlesite_loss(etab, E_idx, S, mask, vocab, weight=0.1, from_val=False):
    ref_seqs = S
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)

    isnt_x_aa = (torch.logical_and(ref_seqs != 20, ref_seqs != 21)).float() # b x L
    full_mask = mask * isnt_x_aa

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = torch.diagonal(etab[:, :, 0:1].squeeze(2), dim1=-2, dim2=-1) # b x L x 22
    pair_etab = etab[:, :, 1:] # b x L x 29 x 22 x 22
    # E_idx for all but self
    E_idx_jn = E_idx[:, :, 1:] # b x L x 29
    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx_jn) # b x L x 29
    # expand the matrix so we can gather pair energies
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1, 1]).expand(-1, -1, -1, 21, -1) # b x L x 29 x 22 x 1
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1) # b x L x 29 x 22
    # sum_(u != n,m) E_p(a_i,n; A_u)
    sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2) # b x L x 22

    composite_logits = self_etab + sum_pair_nrgs_jn
    log_probs = torch.log_softmax(-composite_logits, dim=-1) # b x L x 22

    if from_val:
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
        ).view(S.size())
        S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
        true_false = (S == S_argmaxed).float()
        loss_av = torch.sum(loss * full_mask) / torch.sum(full_mask)
        return loss, loss_av, true_false
    else:
        S_onehot = torch.nn.functional.one_hot(S, vocab).float()

        # Label smoothing
        S_onehot = S_onehot + weight / float(S_onehot.size(-1))
        S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

        loss = -(S_onehot * log_probs).sum(-1)
        loss_av = torch.sum(loss * full_mask) / 2000.0 #fixed 
    return loss, loss_av


class ESMCContrastiveLoss(nn.Module):
    """InfoNCE-style loss on sequence-level ESM-C embeddings.

    The positive set is MSA embeddings for the current protein; negatives are
    random-sequence embeddings for the same protein and (optionally) MSA
    embeddings from other proteins.
    """
    def __init__(
        self,
        aa_embedding_table,
        embedding_lookup=None,
        temperature=0.07,
        gumbel_tau=1.0,
        num_random_negatives=16,
        num_real_negatives_max=16,
        real_neg_warmup_epochs=50,
        require_same_length_real_negatives=True,
    ):
        super().__init__()
        self.register_buffer("aa_embedding_table", aa_embedding_table.float())
        self.embedding_lookup = embedding_lookup or {}
        self.temperature = temperature
        self.gumbel_tau = gumbel_tau
        self.num_random_negatives = num_random_negatives
        self.num_real_negatives_max = num_real_negatives_max
        self.real_neg_warmup_epochs = max(real_neg_warmup_epochs, 1)
        self.require_same_length_real_negatives = require_same_length_real_negatives

        # Global MSA pool used for "real" negatives from other proteins.
        self._all_real = {}
        self._seq_len_by_name = {}
        self._all_real_by_length = {}
        for name, vals in self.embedding_lookup.items():
            if not isinstance(vals, dict):
                continue
            msa = vals.get('msa')
            if msa is None:
                continue
            msa = torch.as_tensor(msa, dtype=torch.float32)
            if msa.dim() == 1:
                msa = msa.unsqueeze(0)
            if msa.numel() == 0:
                continue
            self._all_real[name] = msa

            seq_len = vals.get('length', vals.get('seq_len'))
            if seq_len is not None:
                seq_len = int(seq_len)
                self._seq_len_by_name[name] = seq_len
                self._all_real_by_length.setdefault(seq_len, []).append(name)

        if (
            self.require_same_length_real_negatives
            and len(self._all_real_by_length) == 0
            and len(self.embedding_lookup) > 0
        ):
            raise ValueError(
                "No sequence lengths found in embedding_lookup for same-length real negatives. "
                "Provide per-protein 'length' (or 'seq_len') metadata, or disable "
                "--esmc_same_length_real_negatives."
            )

    def _sample_rows(self, rows, n_rows):
        if rows is None:
            return None
        rows = torch.as_tensor(rows, dtype=torch.float32, device=self.aa_embedding_table.device)
        if rows.dim() == 1:
            rows = rows.unsqueeze(0)
        if rows.shape[0] == 0 or n_rows <= 0:
            return None
        if rows.shape[0] <= n_rows:
            return rows
        idx = torch.randperm(rows.shape[0], device=rows.device)[:n_rows]
        return rows[idx]


    def _candidate_real_negative_names(self, name, seq_len):
        if self.require_same_length_real_negatives:
            if seq_len is None:
                return []
            names = self._all_real_by_length.get(int(seq_len), [])
            return [n for n in names if n != name]
        return [n for n in self._all_real.keys() if n != name]

    def _real_negative_fraction(self, epoch):
        return min(float(epoch) / float(self.real_neg_warmup_epochs), 1.0)

    def forward(self, log_probs, mask, names, epoch, batch_embedding_lookup=None):
        # Differentiable discrete sampling of predicted sequence.
        aa_probs = F.gumbel_softmax(log_probs, tau=self.gumbel_tau, hard=False, dim=-1)
        token_embs = aa_probs @ self.aa_embedding_table

        seq_mask = mask.unsqueeze(-1)
        denom = seq_mask.sum(dim=1).clamp_min(1.0)
        pred_seq_embs = (token_embs * seq_mask).sum(dim=1) / denom
        pred_seq_embs = F.normalize(pred_seq_embs, p=2, dim=-1)

        real_neg_frac = self._real_negative_fraction(epoch)
        n_real_negs = int(round(real_neg_frac * self.num_real_negatives_max))

        losses = []
        for i, name in enumerate(names):
            sample_data = (
                batch_embedding_lookup.get(name)
                if batch_embedding_lookup is not None
                else self.embedding_lookup.get(name)
            )
            seq_len = int(mask[i].sum().item()) if self.require_same_length_real_negatives else None
            if sample_data is None:
                continue

            pos = self._sample_rows(sample_data.get('msa'), self.num_random_negatives)
            rand_neg = self._sample_rows(sample_data.get('random'), self.num_random_negatives)
            if pos is None or rand_neg is None:
                continue

            real_negs = []
            precomputed_real = sample_data.get('real') if isinstance(sample_data, dict) else None
            if precomputed_real is not None:
                picked_real = self._sample_rows(precomputed_real, n_real_negs)
                if picked_real is not None:
                    real_negs.append(picked_real)
            elif n_real_negs > 0:
                other_names = self._candidate_real_negative_names(name, seq_len)
                if len(other_names) > 0:
                    chosen_names = random.choices(other_names, k=n_real_negs)
                    for other_name in chosen_names:
                        picked = self._sample_rows(self._all_real[other_name], 1)
                        if picked is not None:
                            real_negs.append(picked)

            if len(real_negs) > 0:
                real_negs = torch.cat(real_negs, dim=0)
                negatives = torch.cat([rand_neg, real_negs], dim=0)
            else:
                negatives = rand_neg

            pos = F.normalize(pos, p=2, dim=-1)
            negatives = F.normalize(negatives, p=2, dim=-1)

            all_cands = torch.cat([pos, negatives], dim=0)
            sim = pred_seq_embs[i].unsqueeze(0) @ all_cands.transpose(0, 1)
            sim = sim.squeeze(0) / self.temperature

            # Multi-positive InfoNCE: maximize aggregate positive mass.
            pos_logits = sim[:pos.shape[0]]
            loss_i = -(torch.logsumexp(pos_logits, dim=0) - torch.logsumexp(sim, dim=0))
            losses.append(loss_i)

        if len(losses) == 0:
            return torch.zeros((), device=log_probs.device), 0
        return torch.stack(losses).mean(), len(losses)

class PairformerEdgeAlignmentLoss(nn.Module):
    def __init__(self, pred_dim, target_dim=128, proj_dim=128, var_weight=0.01, var_target=1.0, eps=1e-8):
        super().__init__()
        self.proj = nn.Linear(pred_dim, proj_dim, bias=True) if pred_dim != proj_dim else nn.Identity()
        self.target_proj = nn.Linear(target_dim, proj_dim, bias=True) if target_dim != proj_dim else nn.Identity()
        self.var_weight = var_weight
        self.var_target = var_target
        self.eps = eps

    def _variance_term(self, x):
        if x.shape[0] <= 1:
            return x.new_zeros(())
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(self.var_target - std))

    def forward(self, h_E, E_idx, pairformer_z, pairformer_mask, node_mask):
        bsz, n_res, k, _ = h_E.shape
        tgt_dim = pairformer_z.shape[-1]

        gather_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, tgt_dim)
        pairformer_edges = torch.gather(pairformer_z, 2, gather_idx)

        neighbor_mask = torch.gather(node_mask, 1, E_idx)
        edge_valid = node_mask.unsqueeze(-1) * neighbor_mask
        edge_pair_mask = torch.gather(pairformer_mask, 2, E_idx)
        edge_valid = edge_valid * edge_pair_mask

        valid = edge_valid > 0
        if valid.sum().item() == 0:
            return h_E.new_zeros(()), 0

        pred = self.proj(h_E[valid])
        tgt = self.target_proj(pairformer_edges[valid])

        pred = F.normalize(pred, p=2, dim=-1)
        tgt = F.normalize(tgt, p=2, dim=-1)

        cosine_loss = 1.0 - (pred * tgt).sum(dim=-1)
        cosine_loss = cosine_loss.mean()

        var_loss = self._variance_term(pred)
        loss = cosine_loss + self.var_weight * var_loss
        return loss, int(valid.sum().item())


def structure_loss(frames, backbone_4x4, mask, num_frames=1):
    from openfold.utils.loss import backbone_loss_per_frame

    if isinstance(frames, (list, tuple)):
        frame_traj = list(frames)
    elif isinstance(frames, torch.Tensor):
        if frames.shape[0] <= 0:
            return 0, None, -1
        frame_traj = [frames[i] for i in range(frames.shape[0])]
    else:
        return 0, None, -1

    if num_frames <= 0 or num_frames >= len(frame_traj):
        frames_to_use = frame_traj
    else:
        frames_to_use = frame_traj[-num_frames:]

    loss_vals = []
    residue_loss_vals = []
    for cur_frame in frames_to_use:
        cur_loss, cur_residue_loss = backbone_loss_per_frame(backbone_4x4, mask, traj=cur_frame)
        loss_vals.append(cur_loss)
        residue_loss_vals.append(cur_residue_loss.squeeze(0))

    loss = torch.stack(loss_vals).mean(dim=0)
    residue_loss = torch.stack(residue_loss_vals).mean(dim=0)

    # with open('residue_loss.txt', 'w') as f:
    #     for r_loss in residue_loss.flatten().cpu().numpy():
    #         f.write(str(r_loss) + '\n')

    # raise ValueError

    if loss.isnan().item():
        print("STRUCTURE ALARM")
        print(backbone_4x4)
        print(frames)
        print(loss)
        print(residue_loss)
        print(residue_loss.isnan().any().item())
        print(frames.isnan().any().item())
        print(torch.sum(mask))
        return 0, None, -1
        
    return loss, residue_loss, torch.sum(mask)

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def length(v):
        "Return length of a vector."
        sum = 0.0
        for c in v:
            sum += c * c
        return math.sqrt(sum)

def Vector(x, y, z):
        return (x, y, z)

def dot(u, v):
    "Return dot product of two vectors."
    sum = 0.0
    for cu, cv in zip(u, v):
        sum += cu * cv
    return sum

def subtract(u, v):
    "Return difference between two vectors."
    x = u[0] - v[0]
    y = u[1] - v[1]
    z = u[2] - v[2]
    return Vector(x, y, z)

def cross(u, v):
    "Return the cross product of two vectors."
    x = u[1] * v[2] - u[2] * v[1]
    y = u[2] * v[0] - u[0] * v[2]
    z = u[0] * v[1] - u[1] * v[0]
    return Vector(x, y, z)

def angle(v0, v1):
    "Return angle [0..pi] between two vectors."
    cosa = dot(v0, v1) / length(v0) / length(v1)
    cosa = round(cosa, 10)
    if cosa > 1:
        cosa = 1
    if cosa < -1:
        cosa = -1
    return math.acos(cosa)

def calc_local_angle(p0, p1, p2, p3):
    """Return angles of p0-p1 bond and p3-p1 bond in local reference frame of p1-p2 bond."""
    # define vectors and angle to use in updating pos
    v01 = subtract(p0, p1)
    v12 = subtract(p1, p2)
    v0 = cross(v12, v01)
    t1 = cross(v12, v0)
    t2 = cross(v12, t1)

    # define translation and rotation matrices
    v12_norm = np.divide(v12, length(v12))
    t1_norm = np.divide(t1, length(t1))
    t2_norm = np.divide(t2, length(t2))
    matrix_transform = np.transpose(np.array([v12_norm, t1_norm, t2_norm]))
    matrix_transform_inv = np.linalg.inv(matrix_transform)

    # transform to local reference frame and update position, then transform back
    p3_local = np.matmul(matrix_transform_inv, p3)
    p1_local = np.matmul(matrix_transform_inv, p1)
    p0_local = np.matmul(matrix_transform_inv, p0)
    # print('local frame: ', p3_local, p1_local)
    center_local = p1_local
    p3_local_centered = p3_local - center_local
    p0_local_centered = p0_local - center_local
    return np.arctan2(p3_local_centered[2], p3_local_centered[1]), np.arctan2(p0_local_centered[2], p0_local_centered[1])

def extract_rotation_info(R):
    """Extract rotation angle and axis from rotation matrix"""
    trace_R = np.trace(R)
    cos_angle = 0.5*(trace_R - 1)
    sin_angle = 0.5*np.sqrt((3 - trace_R)*(1 + trace_R))
    angle_from_cos = np.arccos(cos_angle)
    angle_from_sin = np.arcsin(sin_angle)
    # print("angles: ", angle_from_cos, angle_from_sin)
    if trace_R == 3:
        return None, None
    elif trace_R == -1:
        e2 = R[0,1] / (np.sqrt((1 + R[0,0]) * (1 + R[1, 1])))
        e3 = R[0,2] / (np.sqrt((1 + R[0,0]) * (1 + R[2, 2])))
        axis = [np.sqrt(0.5*(1 + R[0, 0])), e2*np.sqrt(0.5*(1 + R[1, 1])), e3*np.sqrt(0.5*(1 + R[2, 2]))]
    else:
        axis = np.multiply((1 / np.sqrt((3 - trace_R) * (1 + trace_R))), [R[2, 1] - R[1, 2], R[0,2] - R[2,0], R[1, 0] - R[0, 1]])
    return angle_from_cos, axis

def update_pos(p0, p1, p2, dihedral, bond_length, bond_length_update):
    """Return vertex p0 consistent with old p0, current p1, p2, p3, dihedral angle, and bond length."""
    
    # define vectors and angle to use in updating pos
    v01 = subtract(p0, p1)
    v12 = subtract(p1, p2)
    v0 = cross(v12, v01)
    t1 = cross(v12, v0)
    t2 = cross(v12, t1)
    a1 = angle(v01, v12)

    # define translation and rotation matrices
    v12_norm = np.divide(v12, length(v12))
    t1_norm = np.divide(t1, length(t1))
    t2_norm = np.divide(t2, length(t2))
    matrix_transform = np.transpose(np.array([v12_norm, t1_norm, t2_norm]))
    matrix_transform_inv = np.linalg.inv(matrix_transform)
    R = [[np.cos(dihedral), -1*np.sin(dihedral)], [np.sin(dihedral), np.cos(dihedral)]]

    # transform to local reference frame and update position, then transform back
    p0_local = np.matmul(matrix_transform_inv, p0)
    p1_local = np.matmul(matrix_transform_inv, p1)
    center_offset = np.cos(a1)*bond_length
    center_local = p1_local + [center_offset, 0, 0]
    p0_local_centered = p0_local - center_local
    p0_local_centered[:2] = p0_local_centered[:2] + [-1*np.cos(a1)*bond_length_update, np.sin(a1)*bond_length_update]
    p0_local_centered[1:] = np.matmul(R, p0_local_centered[1:])
    p0_local_updated = p0_local_centered + center_local
    p0_updated = np.matmul(matrix_transform, p0_local_updated)
    return p0_updated

def matrix_update_pos(p0, p1, p2, dihedral):
    """Return matrix of vertices consistent with current dihedral update."""
    p0_translated = p0 - p1
    v = p2 - p1
    v = np.divide(v, length(v))
    cos_d = np.cos(dihedral)
    sin_d = np.sin(dihedral)

    u_x = np.array([[0, -1*v[2], v[1]], [v[2], 0, -1*v[0]], [-1*v[1], v[0],  0]])
    R = np.identity(3)*cos_d + sin_d*u_x + (1 - cos_d)*np.tensordot(v, v, axes=0)
    p0_translated_rotated = p0_translated @ R.T
    p0_rotated = p0_translated_rotated + p1
    translation = p1 - p1 @ R.T
    return p0_rotated, R, translation
    
def generate_noise(flex_type, noise_level, replicate, epoch, X, bond_length_noise_level=0.0, chain_lens=None, mask=None, noise_lim=1):
    """Calculate noise

    Args
    ----
    replicate : int
        Replicate run number for setting seed
    epoch : int
        Epoch number for setting seed
    flex_type : str
        methodology to calculate flex data
    noise_level : float
        std of noise to add
    X : torch.tensor
        Position of backbone residues
        size: num_residues x 4 x 3
    bond_length_noise_level : float
        std of noise to add to bond lengths
    seg_mask_k : Tensor
        list of segment mask ids
    seg_mask_v : Tensor
        list of segment lengths

    Returns
    -------
    noise : noise to be added to backbone atoms
    """
    if mask is None:
        mask = torch.Tensor([])
    expanded_mask = mask.repeat_interleave(4)
    dev = X.device
    if X.is_cuda:
        X = X.cpu()
        expanded_mask = expanded_mask.cpu()
    X = X.numpy()
    expanded_mask = expanded_mask.numpy()
    # expanded_mask = np.repeat(mask, 4)
    size = X.shape
    
    # print('epoch: ', epoch+1)
    # print('replicate: ', replicate)
    seed = (epoch + 1 + 100*replicate)
    # print('seed: ', seed)
    np.random.seed(seed)
    flex_type_copy = copy.deepcopy(flex_type)
    dihedral_updates = None
    if flex_type.find("fixed") == -1 and flex_type.find("torsion") == -1:
        noise = np.random.normal(loc=0, scale=noise_level, size=size)
    elif flex_type.find("fixed") > -1:
        flex_size = (size[0], size[2])
        noise = np.random.normal(loc=0, scale=noise_level, size=flex_size)
        noise = np.repeat(noise, 4)
        noise = np.reshape(noise, size)
    elif flex_type.find("torsion") > -1:
        if flex_type.find("batch") == -1:
            raise Exception(f"flex type {flex_type} must be ran with batch enabled")
        X = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))
        noise = np.zeros(X.shape)
        prev_noise = np.zeros(X.shape)
        start_idx = 0
        atom_clash_check = -1
        dihedral_updates = np.zeros((X.shape[0], ))
        n_chains = len(chain_lens)
        fallback = np.zeros(n_chains)
        for i_chain, c in enumerate(chain_lens):
            c *= 4
            noise_level_multiplier = 1
            base_noise_level_multiplier = 1
            atom_id = start_idx
            all_noises = {}
            if flex_type.find("processive") > -1:
                all_rotation_matrices = {}
                all_translations = {}
            flex_type = flex_type_copy
            if flex_type.find("processive") > -1:
                rotation_matrix = np.identity(3)
                translation = np.zeros(X[0,:].shape)
            while atom_id < start_idx + c - 4:
                if not expanded_mask[atom_id] or not expanded_mask[atom_id + 4]:
                    atom_id += 4
                    continue
                prev_noise = copy.deepcopy(noise)
                p0 = X[atom_id,:] + noise[atom_id, :]
                p1_id = atom_id+1
                p2_id = atom_id+2
                p3_id = atom_id+4
                o_offset = 0
                if atom_id % 4 == 1 or atom_id % 4 == 3:
                    atom_id += 1
                    continue
                elif atom_id % 4 == 2:
                    p1_id = atom_id+2
                    p2_id = atom_id+3
                    o_offset = 1
                p1 = X[p1_id,:] + noise[p1_id, :]
                p2 = X[p2_id,:] + noise[p2_id, :]
                p3 = X[p3_id,:] + noise[p3_id, :]
                frac_before = float((atom_id - start_idx) / c)
                if flex_type.find("neighbors") > -1:
                    nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(X)
                    orig_indices = nbrs.kneighbors([X[p3_id,:]], return_distance=False)
                    orig_indices = np.squeeze(orig_indices)[1:]
                angle_mean = 0
                axes_overlap = 0
                if flex_type.find("processive") > -1:
                    orig_pos_angle, new_pos_angle = calc_local_angle(p3, p1, p2, X[p3_id])
                    angle_difference = np.mod(orig_pos_angle - new_pos_angle, 2*np.pi)
                    if abs(2*np.pi - angle_difference < angle_difference):
                        angle_difference = -1*(2*np.pi - angle_difference)
                    new_X = X[start_idx:start_idx+c] + noise[start_idx:start_idx+c] - p1
                    new_X = X[atom_id:min(atom_id+30, start_idx+c)] + noise[atom_id:min(atom_id+30, start_idx+c)] - p1
                    X_chain = X[start_idx:start_idx+c] - p1
                    X_chain = X[atom_id:min(atom_id+30, start_idx+c)] - p1
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, X_chain.shape[0]*np.ones(X_chain.shape[0])))
                    align_rotation_angle, align_rotation_axis = extract_rotation_info(R.as_matrix())
                    if align_rotation_angle is not None:
                        axes_overlap = np.exp(-0.5*d) * np.dot((p2 - p1), align_rotation_axis) / np.linalg.norm(p2 - p1)
                        if abs(axes_overlap) > 1:
                            axes_overlap = 1 * np.sign(axes_overlap)
                        angle_mean = axes_overlap*align_rotation_angle

                dihedral_update = np.random.normal(loc=angle_mean, scale=noise_level*(1-abs(axes_overlap)), size=1)[0]
                if flex_type.find("max") > -1:
                    dihedral_update = np.sign(dihedral_update) * min(abs(dihedral_update), 0.2)
                
                if atom_id % 4 == 0 or atom_id % 4 == 2:
                    dihedral_update = noise_level_multiplier*dihedral_update
                if frac_before <= 0.5:
                    other_start_idx = start_idx
                    other_end_idx = atom_id + o_offset + 1
                if flex_type.find("processive") > -1 or frac_before > 0.5:
                    other_start_idx = p3_id - (1 - o_offset)
                    other_end_idx = start_idx + c
                dihedral_updates[atom_id] = dihedral_update
                bond_length = length(subtract(p0, p1))
                if flex_type.find("simple") > -1:
                    bond_length_update = np.random.normal(loc=0, scale=bond_length_noise_level, size=1)[0]
                    new_pos = update_pos(p0, p1, p2, -1*dihedral_update, bond_length, bond_length_update)  
                    noise[atom_id, :] = (new_pos - X[atom_id, :])
                    atom_id += 1
                    continue
                prev_noise = copy.deepcopy(noise)
                
                p0_matrix =  X[other_start_idx:other_end_idx, :] + noise[other_start_idx:other_end_idx, :]
                p0_matrix_new, new_rotation_matrix, new_translation = matrix_update_pos(p0_matrix, p1, p2, dihedral_update)
                if flex_type.find("processive") > -1:
                    base_rotation_matrix = copy.deepcopy(rotation_matrix)
                    base_translation = copy.deepcopy(translation)
                    if atom_id == start_idx:
                        rotation_matrix = copy.deepcopy(new_rotation_matrix)
                        translation = copy.deepcopy(new_translation)
                    else:
                        rotation_matrix = (rotation_matrix.T @ new_rotation_matrix.T).T
                        translation = (translation @ new_rotation_matrix.T) + new_translation                   
                
                noise[other_start_idx:other_end_idx, :] = p0_matrix_new - X[other_start_idx:other_end_idx, :]
                atom_step = 1

                if bond_length_noise_level > 0 and flex_type.find("processive") == -1:
                    bond_length_update = np.random.normal(loc=0, scale=noise_level_multiplier*bond_length_noise_level, size=1)[0]
                    new_p0 = X[atom_id,:] + noise[atom_id, :]
                    new_p1 = X[p1_id,:] + noise[p1_id, :]
                    new_p2 = X[p2_id,:] + noise[p2_id, :]
                    bond_length = length(new_p1 - new_p0)
                    new_pos = update_pos(new_p0, new_p1, new_p2, 0, bond_length, bond_length_update) 
                    noise[start_idx:atom_id + o_offset + 1] = np.add(noise[start_idx:atom_id + o_offset + 1], (new_pos - new_p0))

                if flex_type.find('rmsd') > -1:
                    new_X = X[start_idx:start_idx+c] + noise[start_idx:start_idx+c]
                    new_X_displacement = np.mean(new_X, axis=0)
                    new_X = new_X - new_X_displacement
                    X_displacement = np.mean(X[start_idx:start_idx+c], axis=0)
                    X_chain = X[start_idx:start_idx+c] - X_displacement
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, new_X.shape[0]*np.ones(new_X.shape[0])))
                    new_X_rotated = R.apply(new_X)
                    noise[start_idx:start_idx+c] = new_X_rotated - X_chain
                    if d > noise_lim:
                        atom_step = 0
                        noise_level_multiplier /= 1.5
                        noise = prev_noise
                elif flex_type.find("checkpoint") > -1:
                    if flex_type.find("fragments") > -1:
                        start_idx_fragment = max(start_idx, atom_id - 30)
                        start_idx_fragment = atom_id
                        end_idx_fragment = min(start_idx+c, p3_id + 30)
                    else:
                        start_idx_fragment = start_idx
                        end_idx_fragment = start_idx + c
                    new_X = X[start_idx_fragment:end_idx_fragment] + noise[start_idx_fragment:end_idx_fragment]
                    new_X_displacement = np.mean(new_X, axis=0)
                    new_X = new_X - new_X_displacement
                    X_displacement = np.mean(X[start_idx_fragment:end_idx_fragment], axis=0)
                    X_chain = X[start_idx_fragment:end_idx_fragment] - X_displacement
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, new_X.shape[0]*np.ones(new_X.shape[0])))
                    new_X_rotated = R.apply(new_X)
                    d = np.sqrt(np.mean(np.linalg.norm(X_chain - new_X, axis=-1)**2))
                    if d > noise_lim:
                        atom_step = 0
                        noise_level_multiplier /= 1.5
                        noise = prev_noise
                        if noise_level_multiplier < 0.01:
                            if flex_type.find("stepwise") > -1:
                                noises = []
                                steps = list(all_noises.keys())
                                steps.sort()
                                for step in steps:
                                    noise = all_noises[step].reshape(X.shape)
                                    noises.append(noise)
                                return noises, dihedral_updates, fallback
                            flex_type+='_rmsd'
                            flex_type=flex_type.replace("processive_", "")
                            noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                            atom_step = start_idx - atom_id
                            fallback[i_chain] = 1
                elif flex_type.find("neighbors") > -1:
                    new_nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(X + noise)
                    new_indices = new_nbrs.kneighbors([X[p3_id,:] + noise[p3_id,:]], return_distance=False)
                    new_indices = np.squeeze(new_indices)[1:]
                    indices = np.unique(np.concatenate((orig_indices, new_indices), 0))
                    orig_neighbors = (X[indices, :]) - X[p3_id, :]
                    orig_neighbors = np.squeeze(orig_neighbors)
                    orig_distances = np.sqrt(np.sum(orig_neighbors**2, axis=-1))
                    neighbors_LJ_sigma = np.divide(orig_distances, 2**(1/6))
                    neighbors_LJ_A = np.multiply(4,neighbors_LJ_sigma**(12))
                    neighbors_LJ_B = np.multiply(4,neighbors_LJ_sigma**(6))
                    orig_neighbors_energy = -1*np.ones(neighbors_LJ_A.shape)
                    temperature = 300

                    new_neighbors = X[indices, :] + noise[indices, :] - (X[p3_id, :] + noise[p3_id, :])
                    new_distances = np.sqrt(np.sum(new_neighbors**2, axis=-1))
                    new_neighbors_energy = np.divide(neighbors_LJ_A, new_distances**(12)) - np.divide(neighbors_LJ_B, new_distances**(6))
                    accept_prob = np.exp(np.divide(30*(np.sum(orig_neighbors_energy) - np.sum(new_neighbors_energy)), 30*temperature))
                    if np.sum(np.isnan(X + noise)) > 0:
                        atom_step = 0
                        noise = prev_noise
                        noise_level_multiplier /= 1.5
                        if noise_level_multiplier < 0.01:
                            flex_type+='_rmsd'
                            flex_type=flex_type.replace("processive", "")
                            noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                            atom_step = start_idx - atom_id
                            fallback[i_chain] = 1
                    if np.random.rand(1)[0] >= accept_prob:
                        atom_step = 0
                        noise_level_multiplier /= 1.5
                        noise = prev_noise
                        rotation_matrix = base_rotation_matrix
                        translation = base_translation
                        if atom_id == atom_clash_check:
                            base_noise_level_multiplier /= 1.5
                            atom_step = atom_step_backtrack
                            new_atom_id = atom_id + atom_step
                            noise_level_multiplier = base_noise_level_multiplier
                            noise = all_noises[new_atom_id]
                            rotation_matrix = all_rotation_matrices[new_atom_id]
                            translation = all_translations[new_atom_id]
                    else:
                        if atom_id == atom_clash_check:
                            noise_level_multiplier = 1
                            atom_clash_check = -1
                            base_noise_level_multiplier = 1
                        
                    if base_noise_level_multiplier < 0.01:
                        flex_type+='_rmsd'
                        flex_type=flex_type.replace("processive", "")
                        noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                        atom_step = start_idx - atom_id
                        fallback[i_chain] = 1
                    if noise_level_multiplier / base_noise_level_multiplier < 0.1:
                        if atom_id == start_idx:
                            if noise_level_multiplier < 0.00001:
                                flex_type+='_rmsd'
                                flex_type=flex_type.replace("processive", "")
                                noise[start_idx:start_idx+c] = np.zeros(noise[start_idx:start_idx+c].shape)
                                atom_step = start_idx - atom_id
                                fallback[i_chain] = 1
                        else:
                            if atom_id != atom_clash_check:
                                n_iters = 1
                            else:
                                n_iters += 1
                            if flex_type.find("smart") > -1:
                                possible_rotation_axis_points = X[start_idx:atom_id] + noise[start_idx:atom_id]
                                possible_rotation_axis_points = np.delete(possible_rotation_axis_points, slice(3, None, 4), axis=0)
                                atom_id_dists = atom_id - possible_rotation_axis_points
                                atom_id_dists = np.delete(atom_id_dists, slice(1, None, 2), axis=0)
                                possible_rotation_vecs = np.diff(possible_rotation_axis_points, n=1, axis=0)
                                possible_rotation_vecs = np.delete(possible_rotation_axis_points, slice(1, None, 2), axis=0)
                                atom_id_rotation_vec_dists = np.divide(np.linalg.norm(np.cross(possible_rotation_vecs, atom_id_dists), axis=-1), np.linalg.norm(possible_rotation_vecs))
                                max_vec_dist_ind = np.argmax(atom_id_rotation_vec_dists)
                                new_atom_id = int(2*max_vec_dist_ind)
                            else:    
                                if len(dihedral_updates[max(start_idx, atom_id - 20*n_iters):atom_id]) > 0:
                                    new_atom_id = max(start_idx, atom_id - 20*n_iters) + np.argmax(np.array(abs(dihedral_updates[max(start_idx, atom_id - 20*n_iters):atom_id])))
                                else:
                                    new_atom_id = max(start_idx, atom_id - 20*n_iters)
                            noise = all_noises[new_atom_id]
                            rotation_matrix = all_rotation_matrices[new_atom_id]
                            translation = all_translations[new_atom_id]
                            atom_step = new_atom_id - atom_id
                            atom_step_backtrack = copy.deepcopy(atom_step)
                            noise_level_multiplier = base_noise_level_multiplier
                            base_noise_level_multiplier = base_noise_level_multiplier
                            atom_clash_check = atom_id
                                                    
                if atom_step == 1:
                    noise_level_multiplier = base_noise_level_multiplier
                    all_noises[atom_id] = copy.deepcopy(noise)
                    if flex_type.find("processive") > -1:
                        all_rotation_matrices[atom_id] = copy.deepcopy(rotation_matrix)
                        all_translations[atom_id] = copy.deepcopy(translation)
                atom_id += atom_step
                if atom_id == start_idx + c - 5:
                    new_X = X[start_idx:start_idx+c] + noise[start_idx:start_idx+c]
                    new_X_displacement = np.mean(new_X, axis=0)
                    new_X = new_X - new_X_displacement
                    X_displacement = np.mean(X[start_idx:start_idx+c], axis=0)
                    X_chain = X[start_idx:start_idx+c] - X_displacement
                    R, d  = scipy.spatial.transform.Rotation.align_vectors(X_chain, new_X, weights = np.divide(1, new_X.shape[0]*np.ones(new_X.shape[0])))
                    new_X_rotated = R.apply(new_X)
                    noise[start_idx:start_idx+c] = new_X_rotated - X_chain
                
                if flex_type.find("stepwise") > -1 and atom_id > 100:
                    X = X.reshape((int(X.shape[0]/4), 4, 3))
                    noises = []
                    steps = list(all_noises.keys())
                    steps.sort()
                    for step in steps:
                        noise = all_noises[step].reshape(X.shape)
                        noises.append(noise)
                    dihedral_updates = dihedral_updates.reshape((X.shape[0], 4))
                    dihedral_updates = np.swapaxes(dihedral_updates, 0, 1)
                    return noises, dihedral_updates, fallback

            start_idx += c
        X = X.reshape((int(X.shape[0]/4), 4, 3))
        noise = noise.reshape(X.shape)

        # print('noise sum', np.sum(abs(noise)))
        return noise

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., augment_type='atomic', num_chain_embeddings=16, feat_type='protein_mpnn', augment_lim=1.0):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.augment_type = augment_type
        self.augment_lim = augment_lim
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.feat_type = 'protein_mpnn'

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels, all_chain_lens, epoch=None, replicate=None): 
        # if self.augment_type == 'atomic':
        #     X = X + self.augment_eps * np.random.standard_normal(size=np.shape(X))
        # elif self.augment_type.find('torsion') > -1:
        #     for i in range(X.shape[0]):
        #         X[i] += generate_noise(self.augment_type, self.augment_eps, replicate, epoch, X[i], chain_lens=all_chain_lens[i], mask=mask[i])  
        # if self.augment_type == 'atomic': 
        #     if self.training and self.augment_eps > 0:
        #         X = X + self.augment_eps * torch.randn_like(X)
        if self.training and self.augment_eps > 0:
            if self.augment_type.find('torsion') > -1:
                for i in range(X.shape[0]):
                    X[i] += torch.from_numpy(generate_noise(self.augment_type, self.augment_eps, replicate, epoch, X[i], chain_lens=all_chain_lens[i], mask=mask[i], noise_lim=self.augment_lim)).to(device=X.device)
            else:
                X = X + self.augment_eps * torch.randn_like(X)

        if self.feat_type == 'protein_mpnn':
            b = X[:,:,1,:] - X[:,:,0,:]
            c = X[:,:,2,:] - X[:,:,1,:]
            a = torch.cross(b, c, dim=-1)
            Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
            Ca = X[:,:,1,:]
            N = X[:,:,0,:]
            C = X[:,:,2,:]
            O = X[:,:,3,:]
    
            D_neighbors, E_idx = self._dist(Ca, mask)

            RBF_all = []
            RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
            RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
            RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
            RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
            RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
            RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
            RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
            RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
            RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
            RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
            RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
            RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
            RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
            RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
            RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
            RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
            RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
            RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
            RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
            RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
            RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
            RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
            RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
            RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
            RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
            RBF_all = torch.cat(tuple(RBF_all), dim=-1)

            offset = residue_idx[:,:,None]-residue_idx[:,None,:]
            offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

            d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
            E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
            E_positional = self.embeddings(offset.long(), E_chains)
            E = torch.cat((E_positional, RBF_all), -1)
            E = self.edge_embedding(E)
            E = self.norm_edges(E)
        elif self.feat_type == 'coordinator':
            raise ValueError
            featurizer = CoordFeatures(self.edge_features, self.node_features)
            _, E, E_idx = featurizer(X, chain_labels, mask)
        return E, E_idx

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MultiLayerLinear(nn.Module):
    def __init__(self, in_features, out_features, num_layers, activation_layers='relu', dropout=0):
        super(MultiLayerLinear, self).__init__()
        self.activation_layers = activation_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features[i], out_features[i]))
        self.dropout_prob = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout, inplace=False)
            
    def forward(self, x):
        for layer in self.layers:  
            # print('\t', x.shape)
            x = layer(x)
            x = gelu(x)
        if self.dropout_prob > 0:
            x = self.dropout(x)
        return x


class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, output_dim=400, num_encoder_layers=3, num_decoder_layers=3, seq_encoding='one_hot',
        vocab=21, k_neighbors=32, augment_eps=0.1, augment_type='atomic', augment_lim=1.0, dropout=0.1, feat_type='protein_mpnn', use_potts=False, node_self_sub=None, clone=True, struct_predict=False, use_struct_weights=True, multimer_structure_module=False, struct_predict_pairs=True, struct_predict_seq=True, use_seq=True, device='cuda:0' ):
        super(ProteinMPNN, self).__init__()
        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.seq_encoding = seq_encoding
        self.struct_predict = struct_predict
        self.use_struct_weights = use_struct_weights
        self.multimer_structure_module = multimer_structure_module
        self.struct_predict_pairs = struct_predict_pairs
        self.struct_predict_seq = struct_predict_seq
        self.use_seq = use_seq

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, augment_type=augment_type, feat_type=feat_type, augment_lim=augment_lim)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        if self.seq_encoding == 'one_hot':
            self.W_s = nn.Embedding(vocab, hidden_dim)
        elif self.seq_encoding == 'esm2-150M':
            self.W_s = MultiLayerLinear(in_features=[640, 640, 320, 128], out_features=[640, 320, 128, hidden_dim], num_layers=4)
        elif self.seq_encoding == 'esm2-650M':
            self.W_s = MultiLayerLinear(in_features=[1280, 640, 320, 128], out_features=[640, 320, 128, hidden_dim], num_layers=4)
        elif self.seq_encoding == 'esm2-3B':
            self.W_s = MultiLayerLinear(in_features=[2560, 640, 320, 128], out_features=[640, 320, 128, hidden_dim], num_layers=4)
        elif self.seq_encoding == 'esm2-one_hot':
            self.W_s = MultiLayerLinear(in_features=[22, 640, 320, 128], out_features=[640, 320, 128, hidden_dim], num_layers=4)
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        self.use_potts = use_potts
        if self.use_potts:
            self.etab_out = nn.Linear(hidden_dim, output_dim, bias=True)
            self.node_self_sub = node_self_sub
            if self.node_self_sub is not None:
                self.self_E_out = nn.Linear(hidden_dim, 20, bias=True)
            self.clone = clone

        if self.struct_predict: # Load structure module if needed               
            from trunk import FoldingTrunk
            url = "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_structure_module_only_650M.pt"
            model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
            cfg = model_data["cfg"]["model"]
            
            if not self.use_struct_weights and self.multimer_structure_module:
                cfg.trunk['structure_module']['is_multimer'] = True

            if self.use_struct_weights:
                cfg.trunk['use_weights'] = True

            struct_module = FoldingTrunk(**cfg.trunk).to(device)
            if self.use_struct_weights:
                self.node_struct_reshape = nn.Linear(num_letters, 1024)
                self.edge_struct_reshape = nn.Linear(output_dim, 128)
                # self.node_struct_reshape = MultiLayerLinear(in_features=[22, 128, 512], out_features=[128, 512, 1024], num_layers=3, activation_layers='gelu')
                # self.edge_struct_reshape = MultiLayerLinear(in_features=[400, 256, 128], out_features=[256, 128, 128], num_layers=3, activation_layers='gelu')
                module_params = []
                for n, p in struct_module.named_parameters():
                    module_params.append(n)
                
                state_dict = {}
                for param, val in model_data['model'].items():
                    param = param[6:]
                    if '_points' in param:
                        param = '.'.join(param.split('.')[:-1]) + '.linear.' + param.split('.')[-1]
                    if param in module_params:
                        state_dict[param] = val   
                struct_module.load_state_dict(state_dict)
            else:
                print('Not using structure weights')
            
            self.struct_module = struct_module

        for name, p in self.named_parameters():
            if (not name.startswith("struct_module.") and self.use_struct_weights) and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, all_chain_lens=None, epoch=1, replicate=1):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, all_chain_lens, epoch, replicate)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        # Format h_E into Potts model etab
        if self.use_potts:
            if self.clone:
                etab = h_E.clone()
            else:
                etab = h_E
            etab = self.etab_out(etab)
            n_batch, n_res, k, out_dim = etab.shape
            etab = etab * mask.view(n_batch, n_res, 1, 1) # ensure output etab is masked properly
            etab = etab.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
            etab[:, :, 0] = etab[:, :, 0] * torch.eye(20).to(etab.device) # zero off-diagonal energies
            etab = merge_duplicate_pairE(etab, E_idx)
            etab = etab.view(n_batch, n_res, k, out_dim)
            if self.node_self_sub == 'encoder':
                if self.clone:
                    self_E = h_V.clone()
                else:
                    self_E = h_V
                self_E = self.self_E_out(self_E)
                etab[..., 0, :] = torch.diag_embed(self_E, dim1=-2, dim2=-1).flatten(start_dim=-2, end_dim=-1)
        else:
            etab = None

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            if self.use_seq:
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            else:
                h_ESV = h_EXV_encoder
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        if self.use_potts and self.node_self_sub == 'decoder':
            if self.clone:
                self_E = h_V.clone()
            else:
                self_E = h_V
            self_E = self.self_E_out(self_E)
            etab[..., 0, :] = torch.diag_embed(self_E, dim1=-2, dim2=-1).flatten(start_dim=-2, end_dim=-1)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)

        # Struct predict
        if self.struct_predict:
            b, n, k, h = etab.shape
            h2 = int(np.sqrt(h))
            if self.struct_predict_pairs:
                struct_etab = etab.clone().view(b, n, k, h2, h2)
                struct_etab = expand_etab(struct_etab, E_idx).reshape(b, n, n, h).squeeze(-1).to(torch.float32)
            else:
                struct_etab = torch.zeros((b, n, n, h, h)).to(torch.float32)
            if self.struct_predict_seq:
                h_V_fold = logits
            else:
                h_V_fold = torch.zeros_like(logits)
            if self.use_struct_weights:
                h_V_fold = self.node_struct_reshape(logits)
                struct_etab = self.edge_struct_reshape(struct_etab)
            h_V_fold = h_V_fold.to(torch.float32)
            structure = self.struct_module(h_V_fold, struct_etab, 7*torch.ones(mask.shape, dtype=torch.long, device=etab.device), residue_idx.to(torch.long), mask)

            frames = structure['frames']
            positions = structure['positions']
        else:
            frames = None
            positions = None


        return log_probs, etab, E_idx, h_E, frames, positions

    def forward_recycle(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, all_chain_lens=None, epoch=1, replicate=1, num_recycles=3):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, all_chain_lens, epoch, replicate)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend


        for i_cycle in range(num_recycles):
            for layer in self.encoder_layers:
                h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

            # Format h_E into Potts model etab
            if self.clone:
                etab = h_E.clone()
            else:
                etab = h_E
            etab = self.etab_out(etab)
            n_batch, n_res, k, out_dim = etab.shape
            etab = etab * mask.view(n_batch, n_res, 1, 1) # ensure output etab is masked properly
            etab = etab.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
            etab[:, :, 0] = etab[:, :, 0] * torch.eye(20).to(etab.device) # zero off-diagonal energies
            etab = merge_duplicate_pairE(etab, E_idx)
            etab = etab.view(n_batch, n_res, k, out_dim)
            if self.node_self_sub == 'encoder':
                if self.clone:
                    self_E = h_V.clone()
                else:
                    self_E = h_V
                self_E = self.self_E_out(self_E)
                etab[..., 0, :] = torch.diag_embed(self_E, dim1=-2, dim2=-1).flatten(start_dim=-2, end_dim=-1)

            # Concatenate sequence embeddings for autoregressive decoder
            h_S = self.W_s(S)
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            # Build encoder embeddings
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            chain_M = chain_M*mask #update chain_M to include missing regions
            decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)

            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

            if self.node_self_sub == 'decoder':
                if self.clone:
                    self_E = h_V.clone()
                else:
                    self_E = h_V
                self_E = self.self_E_out(self_E)
                etab[..., 0, :] = torch.diag_embed(self_E, dim1=-2, dim2=-1).flatten(start_dim=-2, end_dim=-1)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)

            ## Predict structure and recycle
            # struct_loss, struct_loss_residue = 


        return log_probs, etab, E_idx, h_E

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )
