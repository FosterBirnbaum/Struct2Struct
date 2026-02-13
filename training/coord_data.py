"""Datasets and dataloaders for loading TERMs.

This file contains dataset and dataloader classes
to be used when interacting with TERMs.
"""
import glob
import math
import multiprocessing as mp
import os
import pickle
import random
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import esm as esmlib
from coord_data_utils import extract_knn, get_merge_dups_mask, extract_edge_mapping, get_msa_paired_stats, gather_edges, _esm_featurize, esm_convert, ints_to_seq_torch, esm_ints_to_seq_torch
from noise_utils import generate_noise

# pylint: disable=no-member, not-callable


# Jing featurization functions


def _normalize(tensor, dim=-1):
    '''Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.'''
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.

    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].

    From https://github.com/jingraham/neurips19-graph-protein-design
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    rbf = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return rbf


def _dihedrals(X, eps=1e-7):
    """ Compute dihedral angles between residues given atomic backbone coordinates

    Args
    ----
    X : torch.FloatTensor
        Tensor specifying atomic backbone coordinates
        Shape: num_res x 4 x 3

    Returns
    -------
    D_features : torch.FloatTensor
        Dihedral angles, lifted to the 3-torus
        Shape: num_res x 7
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design

    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index, num_embeddings=16, dev='cpu'):
    """ Sinusoidally encode sequence distances for edges.

    Args
    ----
    edge_index : torch.LongTensor
        Edge indices for sparse representation of protein graph
        Shape: 2 x num_edges
    num_embeddings : int or None, default=128
        Dimensionality of sinusoidal embedding.

    Returns
    -------
    E : torch.FloatTensor
        Sinusoidal encoding of sequence distances
        Shape: num_edges x num_embeddings

    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=dev) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X_ca):
    """ Compute forward and backward vectors per residue.

    Args
    ----
    X_ca : torch.FloatTensor
        Tensor specifying atomic backbone coordinates for CA atoms.
        Shape: num_res x 3

    Returns
    -------
    torch.FloatTensor
        Pairs of forward, backward vectors per residue.
        Shape: num_res x 2 x 3
    """
    # From https://github.com/drorlab/gvp-pytorch
    forward = _normalize(X_ca[1:] - X_ca[:-1])
    backward = _normalize(X_ca[:-1] - X_ca[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    """ Compute vectors pointing in the approximate direction of the sidechain.

    Args
    ----
    X : torch.FloatTensor
        Tensor specifying atomic backbone coordinates.
        Shape: num_res x 4 x 3

    Returns
    -------
    vec : torch.FloatTensor
        Sidechain vectors.
        Shape: num_res x 3
    """
    # From https://github.com/drorlab/gvp-pytorch
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _jing_featurize(protein, dev='cpu'):
    """ Featurize individual proteins for use in torch_geometric Data objects,
    as done in https://github.com/drorlab/gvp-pytorch

    Args
    ----
    protein : dict
        Dictionary of protein features

        - :code:`name` - PDB ID of the protein
        - :code:`coords` - list of dicts specifying backbone atom coordinates
        in the format of that outputted by :code:`parseCoords.py`
        - :code:`seq` - protein sequence
        - :code:`chain_idx` - an integer per residue such that each unique integer represents a unique chain

    Returns
    -------
    torch_geometric.data.Data
        Data object containing
        - :code:`x` - CA atomic coordinates
        - :code:`seq` - sequence of protein
        - :code:`name` - PDB ID of protein
        - :code:`node_s` - Node scalar features
        - :code:`node_v` - Node vector features
        - :code:`edge_s` - Edge scalar features
        - :code:`edge_v` - Edge vector features
        - :code:`edge_index` - Sparse representation of edge
        - :code:`mask` - Residue mask specifying residues with incomplete coordinate sets
    """
    name = protein['name']
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], device=dev, dtype=torch.float32)
        seq = torch.as_tensor(protein['seq'], device=dev, dtype=torch.long)

        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=30, loop=True)  # TODO: make param

        pos_embeddings = _positional_embeddings(edge_index)
        # generate mask for interchain interactions
        pos_chain = (protein['chain_idx'][edge_index.view(-1)]).view(2, -1)
        pos_mask = (pos_chain[0] != pos_chain[1])
        # zero out all interchain positional embeddings
        pos_embeddings = pos_mask.unsqueeze(-1) * pos_embeddings

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=16, device=dev)  # TODO: make param

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(x=X_ca,
                                     seq=seq,
                                     name=name,
                                     node_s=node_s,
                                     node_v=node_v,
                                     edge_s=edge_s,
                                     edge_v=edge_v,
                                     edge_index=edge_index,
                                     mask=mask)
    return data


# Ingraham featurization functions


def _ingraham_featurize(batch, device="cpu", use_sc=False, sc_ids=False, sc_chi=False):
    """ Pack and pad coords in batch into torch tensors
    as done in https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    batch : list of dict
        list of protein backbone coordinate dictionaries,
        in the format of that outputted by :code:`parseCoords.py`
    device : str
        device to place torch tensors

    Returns
    -------
    X : torch.Tensor
        Batched coordinates tensor
    mask : torch.Tensor
        Mask for X
    lengths : np.ndarray
        Array of lengths of batched proteins
    """
    B = len(batch)
    lengths = np.array([b.shape[0] for b in batch], dtype=np.int32)
    l_max = max(lengths)
    n_atoms = 4
    if use_sc: n_atoms = 10
    X = np.zeros([B, l_max, n_atoms, 3])
    if sc_ids: X = np.zeros([B, l_max, n_atoms])
    elif sc_chi: X = -1000*np.ones([B, l_max, n_atoms])
    
    # Build the batch
    for i, x in enumerate(batch):
        l = x.shape[0]
        if not sc_ids and not sc_chi:
            x_pad = np.pad(x, [[0, l_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan, ))
            X[i, :, :, :] = x_pad
        else:
            x_pad = np.pad(x, [[0, l_max - l], [0, 0]], 'constant', constant_values=(np.nan, ))
            X[i, :, :] = x_pad  

    # Mask
    isnan = np.isnan(X)
    if sc_chi:
        mask = np.isfinite(np.sum(X, (2))).astype(np.float32)
        X[isnan] = -1000.
    elif sc_ids:
        mask = np.isfinite(np.sum(X, (2))).astype(np.float32)
        X[isnan] = -1.
    elif use_sc:
        mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
        X[isnan] = -1.
    else:
        mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
        X[isnan] = 0.
    

    # Conversion
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    return X, mask, lengths

def _quaternions(R, eps=1e-10):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4]
    """
    def _R(i, j):
        return R[..., i, j]

    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(
        torch.abs(1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)) + eps)
    signs = torch.sign(torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q


def _orientations_coarse(X, edge_index, eps=1e-6):
    # Pair features

    # Shifted slices of unit vectors
    dX = X[1:, :] - X[:-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2, :]
    u_1 = U[1:-1, :]
    u_0 = U[2:, :]
    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Bond angle calculation
    cosA = -(u_1 * u_0).sum(-1)
    cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
    A = torch.acos(cosA)
    # Angle between normals
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
    # Backbone features
    AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), -1)
    AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

    # Build relative orientations
    o_1 = F.normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
    O = O.view(list(O.shape[:1]) + [9])
    O = F.pad(O, (0, 0, 1, 2), 'constant', 0)

    # DEBUG: Viz [dense] pairwise orientations
    # O = O.view(list(O.shape[:2]) + [3,3])
    # dX = X.unsqueeze(2) - X.unsqueeze(1)
    # dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
    # dU = dU / torch.norm(dU, dim=-1, keepdim=True)
    # dU = (dU + 1.) / 2.
    # plt.imshow(dU.data.numpy()[0])
    # plt.show()
    # print(dX.size(), O.size(), dU.size())
    # exit(0)

    O_pairs = O[edge_index]
    X_pairs = X[edge_index]

    # Re-view as rotation matrices
    O_pairs = O_pairs.view(list(O_pairs.shape[:-1]) + [3,3])

    # Rotate into local reference frames
    dX = X_pairs[0] - X_pairs[1]
    dU = torch.matmul(O_pairs[1], dX.unsqueeze(-1)).squeeze(-1)
    dU = F.normalize(dU, dim=-1)
    R = torch.matmul(O_pairs[1].transpose(-1, -2), O_pairs[0])
    Q = _quaternions(R)

    # Orientation features
    O_features = torch.cat((dU, Q), dim=-1)

    # DEBUG: Viz pairwise orientations
    # IMG = Q[:,:,:,:3]
    # # IMG = dU
    # dU_full = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3).scatter(
    #     2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), IMG
    # )
    # print(dU_full)
    # dU_full = (dU_full + 1.) / 2.
    # plt.imshow(dU_full.data.numpy()[0])
    # plt.show()
    # exit(0)
    # print(Q.sum(), dU.sum(), R.sum())
    return AD_features, O_features

def _ingraham_geometric_featurize(protein, dev='cpu'):
    """ Featurize individual proteins for use in torch_geometric Data objects,
    as done in https://github.com/drorlab/gvp-pytorch

    Args
    ----
    protein : dict
        Dictionary of protein features

        - :code:`name` - PDB ID of the protein
        - :code:`coords` - list of dicts specifying backbone atom coordinates
        in the format of that outputted by :code:`parseCoords.py`
        - :code:`seq` - protein sequence
        - :code:`chain_idx` - an integer per residue such that each unique integer represents a unique chain

    Returns
    -------
    torch_geometric.data.Data
        Data object containing
        - :code:`x` - CA atomic coordinates
        - :code:`seq` - sequence of protein
        - :code:`name` - PDB ID of protein
        - :code:`node_s` - Node scalar features
        - :code:`node_v` - Node vector features
        - :code:`edge_s` - Edge scalar features
        - :code:`edge_v` - Edge vector features
        - :code:`edge_index` - Sparse representation of edge
        - :code:`mask` - Residue mask specifying residues with incomplete coordinate sets
    """
    name = protein['name']
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], device=dev, dtype=torch.float32)
        seq = torch.as_tensor(protein['seq'], device=dev, dtype=torch.long)

        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=30, loop=True)  # TODO: make param

        pos_embeddings = _positional_embeddings(edge_index)
        # generate mask for interchain interactions
        pos_chain = (protein['chain_idx'][edge_index.view(-1)]).view(2, -1)
        pos_mask = (pos_chain[0] != pos_chain[1])
        # zero out all interchain positional embeddings
        pos_embeddings = pos_mask.unsqueeze(-1) * pos_embeddings

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=16, device=dev)  # TODO: make param

        dihedrals = _dihedrals(coords)
        _, orientations = _orientations_coarse(X_ca, edge_index)

        node_features = dihedrals
        edge_features = torch.cat([pos_embeddings, rbf, orientations], dim=-1)

        node_features, edge_features, = map(torch.nan_to_num, (node_features, edge_features))

    data = torch_geometric.data.Data(x=X_ca,
                                     seq=seq,
                                     name=name,
                                     node_features=node_features,
                                     edge_features=edge_features,
                                     edge_index=edge_index,
                                     mask=mask)
    return data


# Batching functions


def convert(tensor):
    """Converts given tensor from numpy to pytorch."""
    return torch.from_numpy(tensor)

def convert_all(data):
    """Convert all tensor from numpy to pytorch"""
    for k, v in data.items():
        if type(v) == np.ndarray:
            data[k] = convert(v)
        else:
            data[k] = v
    return data

def _randomize_inds(length, random_type, pdb_id):
    """Produces randomized order of indices."""
    inds = np.arange(length)
    if random_type.find("fixed") > -1:
        seed = sum([ord(char) for char in pdb_id])
        random.seed(seed)
    random.shuffle(inds)
    other = inds[inds != 0]
    inds = np.insert(other, 0, 0)
    return inds

def _package(b_idx, epoch=0, msa_type="", msa_id_cutoff=0.5, flex_type="", replicate=1, noise_level=0, bond_length_noise_level=0, bond_angle_noise_level=0, noise_lim=0, pair_etab_dir='', esm_model=None):
    """Package the given datapoints into tensors based on provided indices.

    Tensors are extracted from the data and padded. Coordinates are featurized
    and the length of TERMs and chain IDs are added to the data.

    Args
    ----
    b_idx : list of tuples (dicts, int)
        The feature dictionaries, as well as an int for the sum of the lengths of all TERMs,
        for each datapoint to package.

    Returns
    -------
    dict
        Collection of batched features required for running TERMinator. This contains:

        - :code:`msas` - the sequences for each TERM match to the target structure

        - :code:`features` - the :math:`\\phi, \\psi, \\omega`, and environment values of the TERM matches

        - :code:`ppoe` - the :math:`\\phi, \\psi, \\omega`, and environment values of the target structure

        - :code:`seq_lens` - lengths of the target sequences

        - :code:`focuses` - the corresponding target structure residue index for each TERM residue

        - :code:`contact_idxs` - contact indices for each TERM residue

        - :code:`src_key_mask` - mask for TERM residue padding

        - :code:`X` - coordinates

        - :code:`x_mask` - mask for the target structure

        - :code:`seqs` - the target sequences

        - :code:`ids` - the PDB ids

        - :code:`chain_idx` - the chain IDs
    """
    # wrap up all the tensors with proper padding and masks
    batch = [data[0] for data in b_idx]
    focus_lens = [data[1] for data in b_idx]
    features, msas, focuses, seq_lens, coords, sc_coords, sc_ids = [], [], [], [], [], [], []
    
    term_lens = []
    seqs = []
    seq_sims = []
    ids = []
    chain_lens = []
    ppoe = []
    contact_idxs = []
    gvp_data = []
    geometric_data = []

    sortcery_seqs = []
    sortcery_nrgs = []

    evcouplings_etabs = []
    use_evcoupling = False
    use_sc = False

    pair_etabs = []

    esm_embs = []
    esm_attns = []

    for i, data in enumerate(batch):
        # have to transpose these two because then we can use pad_sequence for padding
        features.append(convert(data['features']).transpose(0, 1))
        msas.append(convert(data['msas']).transpose(0, 1))

        ppoe.append(convert(data['ppoe']))
        focuses.append(convert(data['focuses']))
        contact_idxs.append(convert(data['contact_idxs']))
        seq_lens.append(data['seq_len'])
        term_lens.append(data['term_lens'].tolist())

        if flex_type:
            data['coords'] += generate_noise(flex_type, noise_level, data['pdb'], replicate, epoch, data['coords'], bond_length_noise_level=bond_length_noise_level, 
                                             bond_angle_noise_level=bond_angle_noise_level, chain_lens=data['chain_lens'], noise_lim=noise_lim, dtype='numpy')

        coords.append(data['coords'])
        ids.append(data['pdb'])
        chain_lens.append(data['chain_lens'])
        if 'sc_coords' in data.keys():
            use_sc = True
            sc_coords.append(data['sc_coords'])
            sc_ids.append(data['sc_ids'])

        cutoff = 1
        if msa_type and msa_type.find("single") == -1 and msa_type.find("sample") == -1:
            seq_ids = np.array(data['seq_id'])
            inds = np.argsort(seq_ids)
            seq_ids = seq_ids[inds[::-1]]
            cur_seqs = data['sequence'][inds[::-1]]
            num_above_cutoff = np.sum(seq_ids >= msa_id_cutoff)
            seq_ids = seq_ids[seq_ids > msa_id_cutoff]
            cur_seqs = cur_seqs[:num_above_cutoff]
            if msa_type.find("full_sim") > -1:
                try:
                    if msa_type.find("cutoff") > -1:
                        cutoff = float(msa_type.split('_')[-1])
                        num_below_cutoff = seq_ids[seq_ids < cutoff].shape[0]
                        if num_below_cutoff < 20:
                            cutoff = seq_ids[int(np.floor(0.1*len(seq_ids)))]
                        num_above_cutoff = np.sum(seq_ids >= cutoff)
                        seq_ids = seq_ids[seq_ids < cutoff]
                        native_seq = cur_seqs[0]
                        cur_seqs = cur_seqs[num_above_cutoff:]
                        seq_ids = np.insert(seq_ids, 0, 1, axis=0)
                        cur_seqs = np.insert(cur_seqs, 0, native_seq, axis=0)  
                        if msa_type.find("random") > -1 or msa_type.find("fixed") > -1:
                            inds = _randomize_inds(len(seq_ids), msa_type, data['pdb'])
                            cur_seqs = cur_seqs[inds]
                            seq_ids = seq_ids[inds]
                    seqs.append(convert(cur_seqs))
                    seq_sims.append(convert(seq_ids))
                except Exception as e:
                    print(e)
                    seqs.append(convert(data['sequence']))
                    seq_sims.append(None)
            elif (msa_type.find("full_random") > -1 or msa_type.find("full_fixed") > -1):
                try:
                    inds = _randomize_inds(len(seq_ids), msa_type, data['pdb'])
                    cur_seqs = cur_seqs[inds]
                    seq_ids = seq_ids[inds]
                    seqs.append(convert(cur_seqs))
                    seq_sims.append(convert(seq_ids))
                except Exception as e:
                    print(e)
                    seqs.append(convert(data['sequence']))
                    seq_sims.append(None)
            else:
                seqs.append(convert(cur_seqs))
                seq_sims.append(convert(seq_ids))

            # Make pair etab
            if not pair_etab_dir:
                continue
            pair_etab_path = os.path.join(pair_etab_dir, data['pdb'], data['pdb'] + '_' + str(cutoff) + '.pair_etabs')
            if not os.path.exists(pair_etab_path):
                msa = convert(cur_seqs)
                X = np.expand_dims(data['coords'], 0)
                X = torch.from_numpy(X[:,:,1,:])
                mask = torch.ones(X.shape[:-1])
                _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
                E_idx = E_idx[0]
                row_contains_no_pad = ((msa != 21) & (msa != 20)).any(dim=1)
                num_alignments = torch.sum(row_contains_no_pad).item()
                msa = msa[:num_alignments]
                pair_etab = get_msa_paired_stats(msa, E_idx)
                with open(pair_etab_path, 'wb') as f:
                    pickle.dump(pair_etab, f)
            else:
                with open(pair_etab_path, 'rb') as f:
                    pair_etab = pickle.load(f)
            pair_etabs.append(pair_etab)
                
        else:
            seqs.append(convert(data['sequence']))
            if len(data['sequence'].shape) > 1:
                raise ValueError
            seq_sims.append(None)

        if 'sortcery_seqs' in data:
            assert len(batch) == 1, "batch_size for SORTCERY fine-tuning should be set to 1"
            sortcery_seqs = convert(data['sortcery_seqs']).unsqueeze(0)
        if 'sortcery_nrgs' in data:
            sortcery_nrgs = convert(data['sortcery_nrgs']).unsqueeze(0)

        chain_idx = []
        for i, c_len in enumerate(data['chain_lens']):
            chain_idx.append(torch.ones(c_len) * i)
        chain_idx = torch.cat(chain_idx, dim=0)
        gvp_data.append(
            _jing_featurize({
                'name': data['pdb'],
                'coords': data['coords'],
                'seq': data['sequence'],
                'chain_idx': chain_idx
            }))
        geometric_data.append(
            _ingraham_geometric_featurize({
                'name': data['pdb'],
                'coords': data['coords'],
                'seq': data['sequence'],
                'chain_idx': chain_idx
            }))
        
        if 'evcouplings_self_energies' in data and data['evcouplings_self_energies'] is not None:
            X = np.expand_dims(data['coords'], 0)
            X = torch.from_numpy(X[:,:,1,:])
            mask = torch.ones(X.shape[:-1])
            _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
            E_idx = E_idx[0]
            self_energy = convert(data['evcouplings_self_energies']).to(dtype=torch.float32)
            pair_energy = convert(data['evcouplings_pair_energies']).to(dtype=torch.float32)
            E_idx_pair = E_idx[:,1:].unsqueeze(-1).unsqueeze(-1).expand((E_idx.shape[0], E_idx.shape[1]-1, pair_energy.shape[2], pair_energy.shape[2]))
            pair_energy_nn = torch.gather(pair_energy, 1, E_idx_pair)
            self_energy_expand = self_energy.unsqueeze(-1).expand((self_energy.shape[0], self_energy.shape[1], self_energy.shape[1]))
            self_energy_nn = torch.eye(20).unsqueeze(0).expand(self_energy_expand.shape)
            self_energy_nn = torch.multiply(self_energy_nn, self_energy_expand).unsqueeze(1)
            evcouplings_etab = torch.cat([self_energy_nn, pair_energy_nn], 1)
            evcouplings_etabs.append(evcouplings_etab)
            use_evcoupling = True

    # transpose back after padding
    features = pad_sequence(features, batch_first=True).transpose(1, 2)
    msas = pad_sequence(msas, batch_first=True).transpose(1, 2).long()

    # we can pad these using standard pad_sequence
    if use_evcoupling:
        ev_etabs = pad_sequence(evcouplings_etabs, batch_first=True, padding_value=-1)
    else:
        ev_etabs = None
    if len(pair_etabs) > 0:
        pair_etabs = pad_sequence(pair_etabs, batch_first=True, padding_value=-1)
    else:
        pair_etabs = None

    ppoe = pad_sequence(ppoe, batch_first=True)
    focuses = pad_sequence(focuses, batch_first=True)
    contact_idxs = pad_sequence(contact_idxs, batch_first=True)
    src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).bool()
    if msa_type and msa_type.find("full") > -1 and msa_type.find("single") == -1 and msa_type.find("sample") == -1:
        padded_seqs = []
        max_length = 0
        max_depth = 0
        for seq in seqs:
            max_depth = max(max_depth, seq.shape[0])
            max_length = max(max_length, seq.shape[1])
        for seq in seqs:      
            indiv_seqs = list(torch.tensor_split(seq, seq.shape[0], dim=0))
            for i, _ in enumerate(indiv_seqs):
                indiv_seqs[i] = torch.squeeze(indiv_seqs[i])
            indiv_seqs[0] = torch.nn.functional.pad(indiv_seqs[0], (0, max_length - indiv_seqs[0].shape[0]), "constant", 21)
            indiv_seqs = pad_sequence(indiv_seqs, batch_first=True, padding_value = 21)
            indiv_seqs = torch.nn.functional.pad(indiv_seqs, (0, 0, 0, max_depth - indiv_seqs.shape[0]), "constant", 21)
            indiv_seqs = torch.transpose(indiv_seqs, 0, 1)
            padded_seqs.append(indiv_seqs)
        seqs = torch.stack(padded_seqs, dim=0)
    else:
        seqs = pad_sequence(seqs, batch_first=True, padding_value = 21)



    # we do some padding so that tensor reshaping during batchifyTERM works
    # TODO(alex): explain this since I have no idea what's going on
    max_aa = focuses.size(-1)
    for lens in term_lens:
        max_term_len = max(lens)
        diff = max_aa - sum(lens)
        lens += [max_term_len] * (diff // max_term_len)
        lens.append(diff % max_term_len)

    # featurize coordinates same way as ingraham et al
    X, x_mask, _ = _ingraham_featurize(coords)

    # pad with -1 so we can store term_lens in a tensor
    seq_lens = torch.tensor(seq_lens)
    max_all_term_lens = max([len(term) for term in term_lens])
    for i, _ in enumerate(term_lens):
        term_lens[i] += [-1] * (max_all_term_lens - len(term_lens[i]))
    term_lens = torch.tensor(term_lens)

    # generate chain_idx from chain_lens
    chain_idx = []
    for c_lens in chain_lens:
        arrs = []
        for i, chain_len in enumerate(c_lens):
            arrs.append(torch.ones(chain_len) * (i+1))
        chain_idx.append(torch.cat(arrs, dim=-1))
    chain_idx = pad_sequence(chain_idx, batch_first=True)

    # Get mapping for undirected code
    X_ca = X[:,:,1,:]
    _, _, E_idx = extract_knn(X_ca, x_mask, 1E-6, 30)
    mapping, all_num_edges = get_merge_dups_mask(E_idx)
    max_num_edges = max(all_num_edges)
    edge_idx, node_endpoint_idx, node_neighbor_idx = extract_edge_mapping(E_idx, mapping, max_num_edges)
    X_sc, x_mask_sc, sc_ids, sc_mask_full, sc_chi, esm_embs, esm_attns, sc_masks, chain_eos_inds = torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])
    return {
        'msas': msas,
        'features': features.float(),
        'ppoe': ppoe.float(),
        'seq_lens': seq_lens,
        'focuses': focuses,
        'contact_idxs': contact_idxs,
        'src_key_mask': src_key_mask,
        'term_lens': term_lens,
        'X': X,
        'x_mask': x_mask,
        'X_sc': X_sc,
        'x_mask_sc': x_mask_sc,
        'sc_chi': sc_chi,
        'sc_mask_full': sc_mask_full,
        'sc_ids': sc_ids,
        'seqs': seqs,
        'ids': ids,
        'chain_idx': chain_idx,
        'gvp_data': gvp_data,
        'sortcery_seqs': sortcery_seqs,
        'sortcery_nrgs': sortcery_nrgs,
        'geometric_data': geometric_data,
        'mapping': mapping,
        'edge_update_inds': (edge_idx, node_endpoint_idx, node_neighbor_idx),
        'chain_lens': chain_lens,
        'ev_etabs': ev_etabs,
        'pair_etabs': pair_etabs,
        'esm_embs': esm_embs,
        'esm_attns': esm_attns,
        'sc_masks': sc_masks,
        'chain_eos_inds': chain_eos_inds
    }

def _apply_mask(array, per):
    num_rows = array.shape[0]
    num_rows_to_zero = int(per * num_rows)
    rows_to_zero = random.sample(range(num_rows), num_rows_to_zero)
    array[rows_to_zero, ...] = -1
    return array

def _apply_seq_mask(seq, data, chain_mask, sc_mask, sc_mask_rate, base_sc_mask, sc_mask_schedule, mask_neighbors, epoch, rseed, converted_to_esm=False):
    if not converted_to_esm:
        seq = ints_to_seq_torch(seq)
    else:
        seq = esm_ints_to_seq_torch(seq)
    seq = list(seq)
    # if len(sc_mask) == 0:
    #     if sc_mask_schedule:
    #         sc_mask_rate = np.concatenate([np.linspace(base_sc_mask, sc_mask_rate, 50), sc_mask_rate*np.ones(50)])[epoch]
    #         if epoch > 50 and sc_mask_rate == 1 and random.random() > 0.5:
    #             sc_mask_rate = random.random() * sc_mask_rate
    #     num_to_mask = max(1, int(round(sc_mask_rate * len(seq))))
    #     random.seed(rseed)
    #     sc_mask = random.sample(range(len(seq)), num_to_mask)
    for i in sc_mask:
        seq[i] = 'X'
    seq = "".join(seq)
    return seq

def find_interface(X, chain_lens, half=False, inter_cutoff=8):
    # import time
    # t0 = time.time()
    X = np.expand_dims(X, 0)
    # t1 = time.time()
    X = X[:,:,1,:]
    # t2 = time.time()
    dX = (np.expand_dims(X, 1) - np.expand_dims(X, 2))
    # t3 = time.time()
    dX = np.sqrt(np.sum(dX**2, 3))[0]
    # t4 = time.time()
    dX_chains = np.zeros((dX.shape[0], len(chain_lens)))
    # t5 = time.time()
    rl = 0
    for i, l in enumerate(chain_lens):
        mins = np.min(dX[:, rl:rl+l], axis=-1)
        mask = (mins > 0) & (mins < inter_cutoff)
        if half:
            mask[:rl+l] = False
        dX_chains[:, i] = mask
        rl += l
        
    # t6 = time.time()
    inter_mask = np.max(dX_chains, axis=1) 
    # t7 = time.time()
    inter_mask = np.nonzero(inter_mask)[0]
    if len(inter_mask) == 1:
        return inter_mask
    inter_mask = inter_mask.squeeze()
    # t8 = time.time()
    # print(t8 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7)
    # print('????')
    return inter_mask
    
class CustomError(Exception):
    def __init__(self, sc_mask, pdb, chains, mask_neighbors, mask_rate, num_rows, shap, gen_sc_mask):
        self.sc_mask = sc_mask
        self.pdb = pdb
        self.chains = chains
        self.mask_neighbors = mask_neighbors
        self.mask_rate = mask_rate
        self.num_rows = num_rows
        self.shap = shap
        super().__init__(f"Error with sc_mask: {sc_mask} for pdb: {pdb} with chains: {chains}.\nMask_neighbors: {mask_neighbors}, mask_rate: {mask_rate}, num_rows: {num_rows}, shape: {shap}, gen_sc_mask: {gen_sc_mask}")


def _wds_package(b_idx, use_sc=False, sc_mask_rate=0.15, base_sc_mask=0.05, chain_mask=False, sc_mask_schedule=False, sc_info='all', sc_mask=[], mask_neighbors=False, 
                 mask_interface=False, half_interface=True, inter_cutoff=8, sc_screen=False, sc_screen_range=[], sc_noise=0, epoch=0, msa_type="", msa_id_cutoff=0.5, flex_type="", replicate=1, noise_level=0, bond_length_noise_level=0, bond_angle_noise_level=0, noise_lim=0, pair_etab_dir='', esm=None, batch_converter=None, use_esm_attns=False, use_reps=False, post_esm_mask=False, from_rla=False, esm_embed_layer=30, fix_seed=True, connect_chains=False, convert_to_esm=False, one_hot=False):
    """Package the given datapoints into tensors based on provided indices.

    Tensors are extracted from the data and padded. Coordinates are featurized
    and the length of TERMs and chain IDs are added to the data.

    Args
    ----
    b_idx : list of tuples (dicts, int)
        The feature dictionaries, as well as an int for the sum of the lengths of all TERMs,
        for each datapoint to package.

    Returns
    -------
    dict
        Collection of batched features required for running TERMinator. This contains:

        - :code:`msas` - the sequences for each TERM match to the target structure

        - :code:`features` - the :math:`\\phi, \\psi, \\omega`, and environment values of the TERM matches

        - :code:`ppoe` - the :math:`\\phi, \\psi, \\omega`, and environment values of the target structure

        - :code:`seq_lens` - lengths of the target sequences

        - :code:`focuses` - the corresponding target structure residue index for each TERM residue

        - :code:`contact_idxs` - contact indices for each TERM residue

        - :code:`X` - coordinates

        - :code:`x_mask` - mask for the target structure

        - :code:`seqs` - the target sequences

        - :code:`ids` - the PDB ids

        - :code:`chain_idx` - the chain IDs
    """
    # wrap up all the tensors with proper padding and masks
    # import time
    # t0 = time.time()
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    batch = [convert_all(data[0]) for data in b_idx]
    features, msas, focuses, seq_lens, coords, sc_coords, sc_ids, sc_chi, sc_masks = [], [], [], [], [], [], [], [], []
    
    term_lens = []
    seqs = []
    seq_sims = []
    ids = []
    chain_lens = []
    ppoe = []
    contact_idxs = []
    gvp_data = []
    geometric_data = []

    sortcery_seqs = []
    sortcery_nrgs = []

    evcouplings_etabs = []
    use_evcoupling = False
    

    pair_etabs = []

    esm_embs = []
    esm_attns = []
    gen_sc_mask_orig = len(sc_mask) == 0
    sc_mask_orig = copy.deepcopy(sc_mask)
    fix_seed=True
    rev_mask_ind=None
    chain_eos_inds = []
    if len(sc_mask) > 0 and sc_mask[0] < 0:
        rev_mask_ind = copy.deepcopy(sc_mask[0])
    for i, data in enumerate(batch):
        gen_sc_mask = copy.deepcopy(gen_sc_mask_orig)
        sc_mask = copy.deepcopy(sc_mask_orig)
        if convert_to_esm:
            data['sequence'] = convert(np.array(esm_convert(data['sequence'].cpu().numpy()))).to(dtype=data['sequence'].dtype)

        # have to transpose these two because then we can use pad_sequence for padding
        features.append(data['features'].transpose(0, 1))
        msas.append(data['msas'].transpose(0, 1))

        ppoe.append(data['ppoe'])
        focuses.append(data['focuses'])
        contact_idxs.append(data['contact_idxs'])
        seq_lens.append(data['seq_len'])

        if flex_type:
            data['coords'] += generate_noise(flex_type, noise_level, data['pdb'], replicate, epoch, data['coords'], bond_length_noise_level=bond_length_noise_level, 
                                             bond_angle_noise_level=bond_angle_noise_level, chain_lens=data['chain_lens'], noise_lim=noise_lim, dtype='numpy')

        coords.append(data['coords'])
        ids.append(data['pdb'])
        chain_lens.append(data['chain_lens'])
        chain_eos_inds.append(torch.cat([torch.arange(sum(data['chain_lens'])), torch.tensor([-1])], dim=-1))
        if 'sc_coords' in data.keys() and use_sc:
            if mask_interface and len(data['chain_lens']) >= 2:
                sc_mask = list(find_interface(data['coords'], data['chain_lens'], half=half_interface, inter_cutoff=inter_cutoff))
                gen_sc_mask = len(sc_mask) == 0
            if gen_sc_mask:
                if sc_mask_schedule:
                    sc_mask_rate = np.concatenate([np.linspace(base_sc_mask, sc_mask_rate, 50), sc_mask_rate*np.ones(50)])[epoch]
                    if epoch > 50 and sc_mask_rate == 1 and random.random() > 0.75:
                        sc_mask_rate = random.random() * sc_mask_rate
                num_rows = data['sc_coords'].shape[0]
                num_rows_to_zero = int(sc_mask_rate * num_rows)
                if fix_seed: random.seed(epoch + sum([ord(char) for char in data['pdb']]))
                sc_mask = list(random.sample(range(num_rows), num_rows_to_zero))

            # if chain_mask and len(data['chain_ids']) > 1:
            #     random.seed(rseed)
            #     chain_id = random.randint(0,len(data['chain_ids']))
            #     sc_mask = []
            #     for res_i in range(data['res_info']):
            #         if data['res_info'][res_i][0] == data['chain_ids'][chain_id]:
            #             sc_mask.append(sc_mask)

            if mask_neighbors:
                X = np.expand_dims(data['coords'], 0)
                X = torch.from_numpy(X[:,:,1,:])
                mask = torch.ones(X.shape[:-1])
                _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
                neighbors_to_mask = set(sc_mask)
                for i in sc_mask:
                    neighbors_to_mask = neighbors_to_mask.union(set(E_idx[0,i].cpu().numpy()))
                sc_mask = list(neighbors_to_mask)
                    
            if rev_mask_ind is not None:
                sc_mask = list(range(data['coords'].shape[0] + rev_mask_ind, data['coords'].shape[0]))
            if fix_seed:
                random.seed(epoch + sum([ord(char) for char in data['pdb']]))
            rseed = epoch + sum([ord(char) for char in data['pdb']])
            use_sc = True
            if sc_screen:
                if len(sc_screen_range) == 0:
                    sc_screen_range = list(range(data['coords'].shape[0]))
                elif sc_screen_range[0] < 0:
                    sc_screen_range = list(range(data['coords'].shape[0] + sc_screen_range[0], data['coords'].shape[0]))
                for i_pos in sc_screen_range:
                    data_c = copy.deepcopy(data)
                    sc_mask = [i_pos]
                    data_c['sc_coords'][sc_mask, ...] = np.nan
                    data_c['sc_ids'][sc_mask, ...] = np.nan
                    data_c['sc_chi'][sc_mask, ...] = np.nan
                    sc_coords.append(data_c['sc_coords'])
                    sc_ids.append(data_c['sc_ids'])
                    sc_chi.append(data_c['sc_chi'])
                    sc_masks.append(torch.tensor(sc_mask))
                    esm_seq = _apply_seq_mask(copy.deepcopy(data['sequence']), data, chain_mask, sc_mask, sc_mask_rate, base_sc_mask, sc_mask_schedule, mask_neighbors, epoch, rseed=rseed, converted_to_esm=convert_to_esm)
                    esm_emb, esm_attn = _esm_featurize(data['chain_lens'], esm_seq, esm, batch_converter, use_esm_attns, esm_embed_layer, from_rla=from_rla, use_reps=use_reps, connect_chains=connect_chains, one_hot=one_hot)
                    if use_esm_attns:
                        esm_attn = esm_attn.unsqueeze(0)
                        X = np.expand_dims(data['coords'], 0)
                        X = torch.from_numpy(X[:,:,1,:])
                        mask = torch.ones(X.shape[:-1])
                        _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
                        esm_attn = gather_edges(esm_attn, E_idx).squeeze(0)
                        esm_attns.append(esm_attn)
                    if post_esm_mask:
                        esm_emb[sc_mask, ...] = torch.nan
                    esm_embs.append(esm_emb)
            else:
                if esm is not None:
                    esm_seq = _apply_seq_mask(copy.deepcopy(data['sequence']), data, chain_mask, sc_mask, sc_mask_rate, base_sc_mask, sc_mask_schedule, mask_neighbors, epoch, rseed=rseed, converted_to_esm=convert_to_esm)
                    esm_emb, esm_attn = _esm_featurize(data['chain_lens'], esm_seq, esm, batch_converter, use_esm_attns, esm_embed_layer, from_rla=from_rla, use_reps=use_reps, connect_chains=connect_chains, one_hot=one_hot)
                    if use_esm_attns:
                        esm_attn = esm_attn.unsqueeze(0)
                        X = np.expand_dims(data['coords'], 0)
                        X = torch.from_numpy(X[:,:,1,:])
                        mask = torch.ones(X.shape[:-1])
                        _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
                        esm_attn = gather_edges(esm_attn, E_idx).squeeze(0)
                        esm_attns.append(esm_attn)
                    esm_embs.append(esm_emb)

                if not 'sc_chi' in data.keys():
                    data['sc_chi'] = np.nan*np.ones(data['sc_coords'].shape)
                # if chain_mask and len(data['chain_ids']) > 1:
                #     if fix_seed: random.seed(rseed)
                #     chain_id = random.randint(0,len(data['chain_ids']))
                #     sc_mask = []
                #     for res_i in range(data['res_info']):
                #         if data['res_info'][res_i][0] == data['chain_ids'][chain_id]:
                #             sc_mask.append(sc_mask)
                #     sc_mask_rate = 0
                
                random.seed(epoch + sum([ord(char) for char in data['pdb']]))
                coords_noise = np.random.normal(loc=0, scale=sc_noise, size=data['sc_coords'].shape)
                # chi_noise = np.random.normal(loc=0, scale=sc_noise, size=data['sc_chi'].shape)
                data['sc_coords'] += coords_noise
                data['sc_coords'][sc_mask, ...] = np.nan
                data['sc_ids'][sc_mask, ...] = np.nan
                data['sc_chi'][sc_mask, ...] = np.nan
                sc_coords.append(data['sc_coords'])
                sc_ids.append(data['sc_ids'])
                sc_chi.append(data['sc_chi'])

            # else:
            #     mask_per = sc_mask_rate
            #     if sc_mask_schedule:
            #         mask_per = np.concatenate([np.linspace(base_sc_mask, mask_per, 50), mask_per*np.ones(50)])[epoch]
            #         if epoch > 50 and mask_per == 1 and random.random() > 0.75:
            #             mask_per = random.random() * mask_per
            #     if sc_info == 'all':
            #         sc_coords.append(_apply_mask(copy.deepcopy(data['sc_coords']), mask_per))
            #         sc_chi.append(_apply_mask(copy.deepcopy(data['sc_chi']), mask_per))
            #     else:
            #         sc_coords.append(-1*np.ones(data['sc_coords'].shape))
            #         sc_chi.append(-1*np.ones(data['sc_chi'].shape))
            #     sc_ids.append(_apply_mask(copy.deepcopy(data['sc_ids']), mask_per))
                sc_masks.append(torch.tensor(sc_mask))
        else:
            sc_masks.append(torch.tensor(sc_mask))
            use_sc = False

        cutoff = 1
        if msa_type and msa_type.find("single") == -1 and msa_type.find("sample") == -1:
            seq_ids = np.array(data['seq_id'])
            inds = np.argsort(seq_ids)
            seq_ids = seq_ids[inds[::-1]]
            cur_seqs = data['sequence'][inds[::-1]]
            num_above_cutoff = np.sum(seq_ids >= msa_id_cutoff)
            seq_ids = seq_ids[seq_ids > msa_id_cutoff]
            cur_seqs = cur_seqs[:num_above_cutoff]
            if msa_type.find("full_sim") > -1:
                try:
                    if msa_type.find("cutoff") > -1:
                        cutoff = float(msa_type.split('_')[-1])
                        num_below_cutoff = seq_ids[seq_ids < cutoff].shape[0]
                        if num_below_cutoff < 20:
                            cutoff = seq_ids[int(np.floor(0.1*len(seq_ids)))]
                        num_above_cutoff = np.sum(seq_ids >= cutoff)
                        seq_ids = seq_ids[seq_ids < cutoff]
                        native_seq = cur_seqs[0]
                        cur_seqs = cur_seqs[num_above_cutoff:]
                        seq_ids = np.insert(seq_ids, 0, 1, axis=0)
                        cur_seqs = np.insert(cur_seqs, 0, native_seq, axis=0)  
                        if msa_type.find("random") > -1 or msa_type.find("fixed") > -1:
                            inds = _randomize_inds(len(seq_ids), msa_type, data['pdb'])
                            cur_seqs = cur_seqs[inds]
                            seq_ids = seq_ids[inds]
                    seqs.append(convert(cur_seqs))
                    seq_sims.append(convert(seq_ids))
                except Exception as e:
                    print(e)
                    seqs.append(data['sequence'])
                    seq_sims.append(None)
            elif (msa_type.find("full_random") > -1 or msa_type.find("full_fixed") > -1):
                try:
                    inds = _randomize_inds(len(seq_ids), msa_type, data['pdb'])
                    cur_seqs = cur_seqs[inds]
                    seq_ids = seq_ids[inds]
                    seqs.append(cur_seqs)
                    seq_sims.append(convert(seq_ids))
                except Exception as e:
                    print(e)
                    seqs.append(convert(data['sequence']))
                    seq_sims.append(None)
            else:
                seqs.append(cur_seqs)
                seq_sims.append(convert(seq_ids))

            # Make pair etab
            if not pair_etab_dir:
                continue
            pair_etab_path = os.path.join(pair_etab_dir, data['pdb'], data['pdb'] + '_' + str(cutoff) + '.pair_etabs')
            if not os.path.exists(pair_etab_path):
                msa = cur_seqs
                X = np.expand_dims(data['coords'], 0)
                X = torch.from_numpy(X[:,:,1,:])
                mask = torch.ones(X.shape[:-1])
                _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
                E_idx = E_idx[0]
                row_contains_no_pad = ((msa != 21) & (msa != 20)).any(dim=1)
                num_alignments = torch.sum(row_contains_no_pad).item()
                msa = msa[:num_alignments]
                pair_etab = get_msa_paired_stats(msa, E_idx)
                with open(pair_etab_path, 'wb') as f:
                    pickle.dump(pair_etab, f)
            else:
                with open(pair_etab_path, 'rb') as f:
                    pair_etab = pickle.load(f)
            pair_etabs.append(pair_etab)
                
        else:
            if len(data['sequence'].shape) > 1:
                data['sequence'] = data['sequence'][0]
            batch_seq = ints_to_seq_torch(data['sequence'])
            indices = torch.from_numpy(np.asarray([alphabet.index(a) for a in batch_seq], dtype=np.int64)).to(dtype=torch.int64)
            seqs.append(indices)
            seq_sims.append(None)

        if 'sortcery_seqs' in data:
            assert len(batch) == 1, "batch_size for SORTCERY fine-tuning should be set to 1"
            sortcery_seqs = data['sortcery_seqs'].unsqueeze(0)
        if 'sortcery_nrgs' in data:
            sortcery_nrgs = data['sortcery_nrgs'].unsqueeze(0)

        chain_idx = []
        for i, c_len in enumerate(data['chain_lens']):
            chain_idx.append(torch.ones(c_len) * i)
        chain_idx = torch.cat(chain_idx, dim=0)
        
        if 'evcouplings_self_energies' in data and data['evcouplings_self_energies'] is not None:
            X = np.expand_dims(data['coords'], 0)
            X = torch.from_numpy(X[:,:,1,:])
            mask = torch.ones(X.shape[:-1])
            _, _, E_idx = extract_knn(X, mask, eps=1E-6, top_k=30)
            E_idx = E_idx[0]
            self_energy = data['evcouplings_self_energies'].to(dtype=torch.float32)
            pair_energy = data['evcouplings_pair_energies'].to(dtype=torch.float32)
            E_idx_pair = E_idx[:,1:].unsqueeze(-1).unsqueeze(-1).expand((E_idx.shape[0], E_idx.shape[1]-1, pair_energy.shape[2], pair_energy.shape[2]))
            pair_energy_nn = torch.gather(pair_energy, 1, E_idx_pair)
            self_energy_expand = self_energy.unsqueeze(-1).expand((self_energy.shape[0], self_energy.shape[1], self_energy.shape[1]))
            self_energy_nn = torch.eye(20).unsqueeze(0).expand(self_energy_expand.shape)
            self_energy_nn = torch.multiply(self_energy_nn, self_energy_expand).unsqueeze(1)
            evcouplings_etab = torch.cat([self_energy_nn, pair_energy_nn], 1)
            evcouplings_etabs.append(evcouplings_etab)
            use_evcoupling = True

    # transpose back after padding
    features = pad_sequence(features, batch_first=True).transpose(1, 2)
    msas = pad_sequence(msas, batch_first=True).transpose(1, 2).long()

    # we can pad these using standard pad_sequence
    if use_evcoupling:
        ev_etabs = pad_sequence(evcouplings_etabs, batch_first=True, padding_value=-1)
    else:
        ev_etabs = torch.Tensor([])
    if len(pair_etabs) > 0:
        pair_etabs = pad_sequence(pair_etabs, batch_first=True, padding_value=-1)
    else:
        pair_etabs = torch.Tensor([])

    ppoe = pad_sequence(ppoe, batch_first=True)
    focuses = pad_sequence(focuses, batch_first=True)
    contact_idxs = pad_sequence(contact_idxs, batch_first=True)
    chain_eos_inds = pad_sequence(chain_eos_inds, batch_first=True, padding_value=-2)
    if msa_type and msa_type.find("full") > -1 and msa_type.find("single") == -1 and msa_type.find("sample") == -1:
        padded_seqs = []
        max_length = 0
        max_depth = 0
        for seq in seqs:
            max_depth = max(max_depth, seq.shape[0])
            max_length = max(max_length, seq.shape[1])
        for seq in seqs:      
            indiv_seqs = list(torch.tensor_split(seq, seq.shape[0], dim=0))
            for i, _ in enumerate(indiv_seqs):
                indiv_seqs[i] = torch.squeeze(indiv_seqs[i])
            indiv_seqs[0] = torch.nn.functional.pad(indiv_seqs[0], (0, max_length - indiv_seqs[0].shape[0]), "constant", 21)
            indiv_seqs = pad_sequence(indiv_seqs, batch_first=True, padding_value = 21)
            indiv_seqs = torch.nn.functional.pad(indiv_seqs, (0, 0, 0, max_depth - indiv_seqs.shape[0]), "constant", 21)
            indiv_seqs = torch.transpose(indiv_seqs, 0, 1)
            padded_seqs.append(indiv_seqs)
        seqs = torch.stack(padded_seqs, dim=0)
    else:
        seqs = pad_sequence(seqs, batch_first=True, padding_value = 0)
        seqs = seqs.to(dtype=torch.int64)


    # we do some padding so that tensor reshaping during batchifyTERM works
    # TODO(alex): explain this since I have no idea what's going on
    max_aa = focuses.size(-1)

    # featurize coordinates same way as ingraham et al
    X, x_mask, _ = _ingraham_featurize(coords)
    if use_sc:
        X_sc, x_mask_sc, _ = _ingraham_featurize(sc_coords, use_sc=True)
        sc_ids, _, _ = _ingraham_featurize(sc_ids, use_sc=True, sc_ids=True)
        sc_chi, _, _ = _ingraham_featurize(sc_chi, use_sc=True, sc_chi=True)
        sc_mask_full = ((X_sc != -1) * x_mask_sc.unsqueeze(-1).unsqueeze(-1).expand(X_sc.shape)).to(dtype=torch.bool)
        sc_masks = pad_sequence(sc_masks, batch_first=True, padding_value=-1)
        if esm is not None:
            esm_embs = pad_sequence(esm_embs, batch_first=True, padding_value = 0)
            if post_esm_mask:
                esm_embs *= x_mask_sc.unsqueeze(-1)
            if use_esm_attns:
                esm_attns = pad_sequence(esm_attns, batch_first=True, padding_value = 0)
            else:
                esm_attns = torch.Tensor([])
        else:
            esm_embs, esm_attns = torch.Tensor([]), torch.Tensor([])

    else:
        X_sc, x_mask_sc, sc_ids, sc_mask_full, sc_chi, sc_chi_mask = torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    # pad with -1 so we can store term_lens in a tensor
    seq_lens = torch.tensor(seq_lens)


    # generate chain_idx from chain_lens
    chain_idx = []
    B, L_max = x_mask.shape
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)
    for b, c_lens in enumerate(chain_lens):
        arrs = []
        l0 = 0
        l1 = 0
        c = 1
        for i, chain_len in enumerate(c_lens):
            l1 += chain_len
            residue_idx[b, l0:l1] = 100*(c-1)+np.arange(l0, l1)
            mask_self[b, l0:l1, l0:l1] = np.zeros([chain_len, chain_len])
                # raise ValueError(x_mask.shape, L_max, l0, l1, chain_lens, c_lens, e)
            l0 += chain_len
            c += 1
            arrs.append(torch.ones(chain_len) * (i+1))
        chain_idx.append(torch.cat(arrs, dim=-1))
    chain_idx = pad_sequence(chain_idx, batch_first=True)
    # t1 = time.time()
    # print('data time ', t1 - t0)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long)
    # chain_M = torch.ones_like(x_mask)
    chain_M = (1 - x_mask_sc) * x_mask
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32)
    return (X, seqs, x_mask, seq_lens, chain_M, residue_idx, mask_self, chain_idx, chain_lens)
    return {
        'msas': msas,
        'features': features.float(),
        'ppoe': ppoe.float(),
        'seq_lens': seq_lens,
        'focuses': focuses,
        'contact_idxs': contact_idxs,
        'X': X,
        'x_mask': x_mask,
        'X_sc': X_sc,
        'x_mask_sc': x_mask_sc,
        'sc_chi': sc_chi,
        'sc_mask_full': sc_mask_full,
        'sc_ids': sc_ids,
        'seqs': seqs,
        'ids': ids,
        'chain_idx': chain_idx,
        'sortcery_seqs': sortcery_seqs,
        'sortcery_nrgs': sortcery_nrgs,
        'chain_lens': chain_lens,
        'ev_etabs': ev_etabs,
        'pair_etabs': pair_etabs,
        'esm_embs': esm_embs,
        'esm_attns': esm_attns,
        'sc_masks': sc_masks,
        'chain_eos_inds': chain_eos_inds
    }


# Non-lazy data loading functions

def load_file(in_folder, pdb_id, min_protein_len=30):
    """Load the data specified in the proper .features file and return them.
    If the read sequence length is less than :code:`min_protein_len`, instead return None.

    Args
    ----
    in_folder : str
        folder to find TERM file.
    pdb_id : str
        PDB ID to load.
    min_protein_len : int
        minimum cutoff for loading TERM file.

    Returns
    -------
    data : dict
        Data from TERM file (as dict)
    total_term_len : int
        Sum of lengths of all TERMs
    seq_len : int
        Length of protein sequence
    """
    path = f"{in_folder}/{pdb_id}/{pdb_id}.features"
    if not os.path.exists(path):
        print(f'no feat file {path} :(')
        return None
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        seq_len = data['seq_len']
        total_term_length = data['term_lens'].sum()
        if seq_len < min_protein_len or data['sequence'].shape[-1] != data['coords'].shape[0]:
            print("length issue with pdb ", pdb_id)
            return None
    return data, total_term_length, seq_len


class TERMDataset(Dataset):
    """TERM Dataset that loads all feature files into a Pytorch Dataset-like structure.

    Attributes
    ----
    dataset : list
        list of tuples containing features, TERM length, and sequence length
    shuffle_idx : list
        array of indices for the dataset, for shuffling
    """
    def __init__(self, in_folder, pdb_ids=None, min_protein_len=30, num_processes=32):
        """
        Initializes current TERM dataset by reading in feature files.

        Reads in all feature files from the given directory, using multiprocessing
        with the provided number of processes. Stores the features, the TERM length,
        and the sequence length as a tuple representing the data. Can read from PDB ids or
        file paths directly. Uses the given protein length as a cutoff.

        Args
        ----
        in_folder : str
            path to directory containing feature files generated by :code:`scripts/data/preprocessing/generateDataset.py`
        pdb_ids: list, optional
            list of pdbs from `in_folder` to include in the dataset
        min_protein_len: int, default=30
            minimum length of a protein in the dataset
        num_processes: int, default=32
            number of processes to use during dataloading
        """
        self.dataset = []
        with mp.Pool(num_processes) as pool:

            if pdb_ids:
                print("Loading feature files")
                progress = tqdm(total=len(pdb_ids))

                def update_progress(res):
                    del res
                    progress.update(1)

                res_list = [
                    pool.apply_async(load_file, (in_folder, id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        features, total_term_length, seq_len = data
                        self.dataset.append((features, total_term_length, seq_len))
            else:
                print("Loading feature file paths")

                filelist = list(glob.glob(f'{in_folder}/*/*.features'))
                progress = tqdm(total=len(filelist))

                def update_progress(res):
                    del res
                    progress.update(1)

                # get pdb_ids
                pdb_ids = [os.path.basename(path)[:-len(".features")] for path in filelist]

                res_list = [
                    pool.apply_async(load_file, (in_folder, id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        features, total_term_length, seq_len = data
                        self.dataset.append((features, total_term_length, seq_len))

            self.shuffle_idx = np.arange(len(self.dataset))

    def shuffle(self):
        """Shuffle the current dataset."""
        np.random.shuffle(self.shuffle_idx)

    def __len__(self):
        """Returns length of the given dataset.

        Returns
        -------
        int
            length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Extract a given item with provided index.

        Args
        ----
        idx : int
            Index of item to return.
        Returns
        ----
        data : dict
            Data from TERM file (as dict)
        total_term_len : int
            Sum of lengths of all TERMs
        seq_len : int
            Length of protein sequence
        """
        data_idx = self.shuffle_idx[idx]
        if isinstance(data_idx, list):
            return [self.dataset[i] for i in data_idx]
        return self.dataset[data_idx]



def TERMBatchSampler_init(self,
                ddp,
                dataset,
                dev,
                batch_size=4,
                sort_data=False,
                shuffle=True,
                semi_shuffle=False,
                semi_shuffle_cluster_size=500,
                batch_shuffle=True,
                drop_last=False,
                max_term_res=55000,
                max_seq_tokens=None,
                msa_type='',
                msa_id_cutoff=0.5,
                flex_type='',
                replicate=1,
                noise_level=0,
                bond_length_noise_level=0,
                bond_angle_noise_level=0,
                noise_lim=0,
                pair_etab_dir=''):
    """
    Reads in and processes a given dataset.

    Given the provided dataset, load all the data. Then cluster the data using
    the provided method, either shuffled or sorted and then shuffled.

    Args
    ----
    dataset : TERMDataset
        Dataset to batch.
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_term_res : int or None, default=55000
        When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
    max_seq_tokens : int or None, default=None
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`. Exactly one of :code:`max_term_res`
        and :code:`max_seq_tokens` must be None.
    """
    if ddp:
        DistributedSampler.__init__(self, dataset)
    else:
        Sampler.__init__(self, dataset)
    self.size = len(dataset)
    self.dataset, self.total_term_lengths, self.seq_lengths = zip(*dataset)
    assert not (max_term_res is None
                and max_seq_tokens is None), "Exactly one of max_term_res and max_seq_tokens must be None"
    if max_term_res is None and max_seq_tokens > 0:
        self.lengths = self.seq_lengths
    elif max_term_res > 0 and max_seq_tokens is None:
        self.lengths = self.total_term_lengths
    else:
        raise ValueError("Exactly one of max_term_res and max_seq_tokens must be None")
    self.shuffle = shuffle
    self.sort_data = sort_data
    self.batch_shuffle = batch_shuffle
    self.batch_size = batch_size
    self.drop_last = drop_last
    self.max_term_res = max_term_res
    self.max_seq_tokens = max_seq_tokens
    self.semi_shuffle = semi_shuffle
    self.semi_shuffle_cluster_size = semi_shuffle_cluster_size
    self.ddp = ddp
    self.msa_type = msa_type
    self.msa_id_cutoff = msa_id_cutoff
    self.dev = dev
    self.epoch = -1
    self.flex_type = flex_type
    self.replicate = replicate
    self.noise_level = noise_level
    self.bond_length_noise_level = bond_length_noise_level
    self.bond_angle_noise_level = bond_angle_noise_level
    self.noise_lim = noise_lim
    self.pair_etab_dir = pair_etab_dir
    assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"

    # initialize clusters
    self._cluster()

def TERMBatchSampler_set_epoch(self, epoch):
    """Set current epoch"""
    self.epoch = epoch

def TERMBatchSampler_cluster(self):
    """ Shuffle data and make clusters of indices corresponding to batches of data.

    This method speeds up training by sorting data points with similar TERM lengths
    together, if :code:`sort_data` or :code:`semi_shuffle` are on. Under `sort_data`,
    the data is sorted by length. Under `semi_shuffle`, the data is broken up
    into clusters based on length and shuffled within the clusters. Otherwise,
    it is randomly shuffled. Data is then loaded into batches based on the number
    of proteins that will fit into the GPU without overloading it, based on
    :code:`max_term_res` or :code:`max_seq_tokens`.
    """
    if self.ddp and dist.get_rank() > 0:
        dims = [0, 0]
        dist.broadcast_object_list(dims, src=0, device=torch.device(self.dev))
        clusters = [list(-1*np.ones(dims[1], dtype=int))]*dims[0]
        dist.broadcast_object_list(clusters, src=0, device=torch.device(self.dev))
        final_clusters = []
        for batch in clusters:
            if batch[0] > 0:
                final_clusters.append(batch[1:batch[0]+1])
        self.clusters = final_clusters
    else:
        # if we sort data, use sorted indexes instead
        if self.sort_data:
            idx_list = np.argsort(self.lengths)
        elif self.semi_shuffle:
            # trying to speed up training
            # by shuffling points with similar term res together
            idx_list = np.argsort(self.lengths)
            shuffle_borders = []

            # break up datapoints into large clusters
            border = 0
            while border < len(self.lengths):
                shuffle_borders.append(border)
                border += self.semi_shuffle_cluster_size

            # shuffle datapoints within clusters
            last_cluster_idx = len(shuffle_borders) - 1
            for cluster_idx in range(last_cluster_idx + 1):
                start = shuffle_borders[cluster_idx]
                if cluster_idx < last_cluster_idx:
                    end = shuffle_borders[cluster_idx + 1]
                    np.random.shuffle(idx_list[start:end])
                else:
                    np.random.shuffle(idx_list[start:])

        else:
            idx_list = list(range(len(self.dataset)))
            np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.batch_size is None:
            if self.max_term_res is None and self.max_seq_tokens > 0:
                cap_len = self.max_seq_tokens
            elif self.max_term_res > 0 and self.max_seq_tokens is None:
                cap_len = self.max_term_res

            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(idx_list):
                current_batch_lens.append(self.lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > cap_len:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.lengths[idx]]
                else:
                    batch.append(idx)

        else:  # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters
        if self.ddp:
            send_clusters = []
            max_len = 0
            for batch in clusters:
                batch = [len(batch)] + batch
                send_clusters.append(batch)
                max_len = max(max_len, len(batch))
            for i, batch in enumerate(send_clusters):
                send_clusters[i] = batch + list(-1*np.ones(max_len - len(batch), dtype=int))
            dims = [len(send_clusters), len(send_clusters[0])]
            dist.broadcast_object_list(dims, src=0, device=torch.device(self.dev))
            dist.broadcast_object_list(send_clusters, src=0, device=torch.device(self.dev))

def TERMBatchSampler_package(self, b_idx):
    """Package the given datapoints into tensors based on provided indices.

    Tensors are extracted from the data and padded. Coordinates are featurized
    and the length of TERMs and chain IDs are added to the data.

    Args
    ----
    b_idx : list of tuples (dicts, int, int)
        The feature dictionaries, the sum of the lengths of all TERMs, and the sum of all sequence lengths
        for each datapoint to package.

    Returns
    -------
    dict
        Collection of batched features required for running TERMinator. This contains:

        - :code:`msas` - the sequences for each TERM match to the target structure

        - :code:`features` - the :math:`\\phi, \\psi, \\omega`, and environment values of the TERM matches

        - :code:`ppoe` - the :math:`\\phi, \\psi, \\omega`, and environment values of the target structure

        - :code:`seq_lens` - lengths of the target sequences

        - :code:`focuses` - the corresponding target structure residue index for each TERM residue

        - :code:`contact_idxs` - contact indices for each TERM residue

        - :code:`src_key_mask` - mask for TERM residue padding

        - :code:`X` - coordinates

        - :code:`x_mask` - mask for the target structure

        - :code:`seqs` - the target sequences

        - :code:`ids` - the PDB ids

        - :code:`chain_idx` - the chain IDs
    """
    if self.msa_type:
        if self.msa_type.find("sample") > -1:
            for data in b_idx:
                seq_ids = np.array(data[0]['seq_id'])
                inds = np.argsort(seq_ids)
                seq_ids = seq_ids[inds[::-1]]
                cur_seqs = data[0]['sequence'][inds[::-1]]
                native_seq = cur_seqs[0]
                num_above_cutoff = np.sum(seq_ids >= 0.5)
                seq_ids = seq_ids[seq_ids > 0.5]
                cur_seqs = cur_seqs[:num_above_cutoff]
                if self.msa_type.find("cutoff") > -1:
                    cutoff = float(self.msa_type.split('_')[-1])
                    num_above_cutoff = np.sum(seq_ids >= cutoff)
                    seq_ids = seq_ids[seq_ids < cutoff]
                    cur_seqs = cur_seqs[num_above_cutoff:]
                num_possible_sequences = len(cur_seqs)
                if num_possible_sequences == 0:
                    data[0]['sequence'] = np.array(native_seq, dtype=int)
                else:
                    randomly_selected_sequence = random.randint(0, num_possible_sequences-1)
                    data[0]['sequence'] = np.array(cur_seqs[randomly_selected_sequence], dtype=int)
        elif self.msa_type.find("single") > -1:
            for data in b_idx:
                seq_ids = np.array(data[0]['seq_id'])
                inds = np.argsort(seq_ids)
                cur_seqs = data[0]['sequence'][inds[::-1]]
                data[0]['sequence'] = np.array(cur_seqs[0], dtype=int)
                seq_ids = seq_ids[inds[::-1]]
                data[0]['seq_id'] = np.array(seq_ids[0], dtype=float)
    return _package([b[0:2] for b in b_idx], self.epoch, self.msa_type, self.msa_id_cutoff, self.flex_type, self.replicate, self.noise_level, self.bond_length_noise_level,
                    self.bond_angle_noise_level, self.noise_lim, self.pair_etab_dir)

def TERMBatchSampler_len(self):
    """Returns length of dataset, i.e. number of batches.

    Returns
    -------
    int
        length of dataset.
    """
    if not self.ddp:
        return len(self.clusters)
    if len(self.clusters) % self.num_replicas != 0:
        if not self.drop_last:
            return math.floor((len(self.clusters) + self.num_replicas) / self.num_replicas)
        else:
            return math.floor((len(self.clusters)) / self.num_replicas)
    return math.ceil(len(self.clusters) / self.num_replicas)

def TERMBatchSampler_iter(self):
    """Allows iteration over dataset."""
    if not self.ddp:
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch
    else:
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.clusters), generator=g).tolist()
        else:
            indices = list(range(len(self.clusters)))
        
        self.num_samples = self.__len__()
        self.total_size = self.num_samples * self.num_replicas
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        for index in indices:
            yield self.clusters[index]


class TERMBatchSamplerWrapper(object):
    """BatchSampler/Dataloader helper class for TERM data using TERMDataset.

    Attributes in wrapped TERMBatchSampler class
    ----
    size: int
        Length of the dataset
    dataset: List
        List of features from TERM dataset
    total_term_lengths: List
        List of TERM lengths from the given dataset
    seq_lengths: List
        List of sequence lengths from the given dataset
    lengths: List
        TERM lengths or sequence lengths, depending on
        whether :code:`max_term_res` or :code:`max_seq_tokens` is set.
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_term_res : int or None, default=55000
        When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
    max_seq_tokens : int or None, default=None
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.
    """
    def __init__(self,
                 ddp):
        self.ddp = ddp
        if ddp:
            print('using distributed sampler!')
            class TERMBatchSampler(DistributedSampler):
                pass
        else:
            class TERMBatchSampler(Sampler):
                pass
        TERMBatchSampler.__init__ = TERMBatchSampler_init
        TERMBatchSampler._cluster = TERMBatchSampler_cluster
        TERMBatchSampler.package = TERMBatchSampler_package
        TERMBatchSampler.__len__ = TERMBatchSampler_len
        TERMBatchSampler.__iter__ = TERMBatchSampler_iter
        TERMBatchSampler._set_epoch = TERMBatchSampler_set_epoch
        self.sampler = TERMBatchSampler


def WDSBatchSampler_init(self,
                ddp,
                dataset,
                batch_size=4,
                sort_data=False,
                shuffle=True,
                semi_shuffle=False,
                semi_shuffle_cluster_size=500,
                batch_shuffle=True,
                drop_last=False,
                max_term_res=55000,
                max_seq_tokens=None,
                msa_type='',
                msa_id_cutoff=0.5,
                flex_type='',
                replicate=1,
                noise_level=0,
                bond_length_noise_level=0,
                bond_angle_noise_level=0,
                noise_lim=0,
                pair_etab_dir='',
                dev='cpu',
                use_sc=False,
                sc_mask_rate=0.15,
                base_sc_mask=0.05,
                chain_mask=False,
                sc_mask_schedule=False,
                sc_info='full',
                sc_mask=[],
                sc_noise=0,
                mask_neighbors=False,
                mask_interface=False,
                half_interface=True,
                inter_cutoff=8,
                sc_screen=False,
                sc_screen_range=[],
                esm=None,
                batch_converter=None,
                use_esm_attns=False,
                use_reps=False,
                post_esm_mask=False,
                from_rla=False,
                esm_embed_layer=30,
                fix_seed=True,
                connect_chains=False,
                convert_to_esm=False,
                one_hot=False):
    """
    Reads in and processes a given dataset.

    Given the provided dataset, load all the data. Then cluster the data using
    the provided method, either shuffled or sorted and then shuffled.

    Args
    ----
    dataset : TERMDataset
        Dataset to batch.
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_term_res : int or None, default=55000
        When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
    max_seq_tokens : int or None, default=None
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`. Exactly one of :code:`max_term_res`
        and :code:`max_seq_tokens` must be None.
    TODO
    """
    if ddp:
        DistributedSampler.__init__(self, dataset)
    else:
        Sampler.__init__(self, dataset)
    self.dataset = dataset
    self.seq_lengths = []
    for i, data in enumerate(self.dataset):
        self.seq_lengths.append(data[0]['coords'].shape[0])
    self.size=len(self.dataset)
    self.num_items = len(self.dataset)
    assert not (max_term_res is None
                and max_seq_tokens is None), "Exactly one of max_term_res and max_seq_tokens must be None"
    if max_term_res is None and max_seq_tokens > 0:
        self.lengths = self.seq_lengths
    elif max_term_res > 0 and max_seq_tokens is None:
        self.lengths = self.seq_lengths
    else:
        raise ValueError("Exactly one of max_term_res and max_seq_tokens must be None")
    self.shuffle = shuffle
    self.sort_data = sort_data
    self.batch_shuffle = batch_shuffle
    self.batch_size = batch_size
    self.drop_last = drop_last
    self.max_term_res = max_term_res
    self.max_seq_tokens = max_seq_tokens
    self.semi_shuffle = semi_shuffle
    self.semi_shuffle_cluster_size = semi_shuffle_cluster_size
    self.ddp = ddp
    self.msa_type = msa_type
    self.msa_id_cutoff = msa_id_cutoff
    self.dev = dev
    self.epoch = -1
    self.flex_type = flex_type
    self.replicate = replicate
    self.noise_level = noise_level
    self.bond_length_noise_level = bond_length_noise_level
    self.bond_angle_noise_level = bond_angle_noise_level
    self.noise_lim = noise_lim
    self.pair_etab_dir = pair_etab_dir
    self.use_sc = use_sc
    self.sc_mask_rate = sc_mask_rate
    self.sc_mask = sc_mask
    self.base_sc_mask = base_sc_mask
    self.chain_mask = chain_mask
    self.sc_mask_schedule = sc_mask_schedule
    self.sc_info = sc_info
    self.sc_noise = sc_noise
    self.mask_neighbors = mask_neighbors
    self.mask_interface = mask_interface
    self.half_interface = half_interface
    self.inter_cutoff = inter_cutoff
    self.sc_screen = sc_screen
    self.sc_screen_range = sc_screen_range
    self.esm = esm
    self.batch_converter = batch_converter
    self.use_esm_attns = use_esm_attns
    self.use_reps = use_reps
    self.post_esm_mask = post_esm_mask
    self.from_rla = from_rla
    self.esm_embed_layer = esm_embed_layer
    self.fix_seed = fix_seed
    self.connect_chains = connect_chains
    self.convert_to_esm = convert_to_esm
    self.one_hot = one_hot
    assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"

    # initialize clusters
    self._cluster()

def WDSBatchSampler_set_epoch(self, epoch):
    """Set current epoch"""
    self.epoch = epoch

def WDSBatchSampler_cluster(self):
    """ Shuffle data and make clusters of indices corresponding to batches of data.

    This method speeds up training by sorting data points with similar TERM lengths
    together, if :code:`sort_data` or :code:`semi_shuffle` are on. Under `sort_data`,
    the data is sorted by length. Under `semi_shuffle`, the data is broken up
    into clusters based on length and shuffled within the clusters. Otherwise,
    it is randomly shuffled. Data is then loaded into batches based on the number
    of proteins that will fit into the GPU without overloading it, based on
    :code:`max_term_res` or :code:`max_seq_tokens`.
    """
    if self.ddp and dist.get_rank() > 0:
        dims = [0, 0]
        dist.broadcast_object_list(dims, src=0, device=torch.device(self.dev))
        clusters = [list(-1*np.ones(dims[1], dtype=int))]*dims[0]
        dist.broadcast_object_list(clusters, src=0, device=torch.device(self.dev))
        final_clusters = []
        for batch in clusters:
            if batch[0] > 0:
                final_clusters.append(batch[1:batch[0]+1])
        self.clusters = final_clusters
    else:
        # if we sort data, use sorted indexes instead
        if self.sort_data:
            idx_list = np.argsort(self.lengths)
        elif self.semi_shuffle:
            # trying to speed up training
            # by shuffling points with similar term res together
            idx_list = np.argsort(self.lengths)
            shuffle_borders = []

            # break up datapoints into large clusters
            border = 0
            while border < len(self.lengths):
                shuffle_borders.append(border)
                border += self.semi_shuffle_cluster_size

            # shuffle datapoints within clusters
            last_cluster_idx = len(shuffle_borders) - 1
            for cluster_idx in range(last_cluster_idx + 1):
                start = shuffle_borders[cluster_idx]
                if cluster_idx < last_cluster_idx:
                    end = shuffle_borders[cluster_idx + 1]
                    np.random.shuffle(idx_list[start:end])
                else:
                    np.random.shuffle(idx_list[start:])

        else:
            idx_list = list(range(self.num_items))
            np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.batch_size is None:
            if self.max_term_res is None and self.max_seq_tokens > 0:
                cap_len = self.max_seq_tokens
            elif self.max_term_res > 0 and self.max_seq_tokens is None:
                cap_len = self.max_term_res

            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(idx_list):
                current_batch_lens.append(self.lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > cap_len:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.lengths[idx]]
                else:
                    batch.append(idx)

        else:  # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters
        if self.ddp:
            send_clusters = []
            max_len = 0
            for batch in clusters:
                batch = [len(batch)] + batch
                send_clusters.append(batch)
                max_len = max(max_len, len(batch))
            for i, batch in enumerate(send_clusters):
                send_clusters[i] = batch + list(-1*np.ones(max_len - len(batch), dtype=int))
            dims = [len(send_clusters), len(send_clusters[0])]
            dist.broadcast_object_list(dims, src=0, device=torch.device(self.dev))
            dist.broadcast_object_list(send_clusters, src=0, device=torch.device(self.dev))

def WDSBatchSampler_package(self, b_idx):
    """Package the given datapoints into tensors based on provided indices.

    Tensors are extracted from the data and padded. Coordinates are featurized
    and the length of TERMs and chain IDs are added to the data.

    Args
    ----
    b_idx : list of tuples (dicts, int, int)
        The feature dictionaries, the sum of the lengths of all TERMs, and the sum of all sequence lengths
        for each datapoint to package.

    Returns
    -------
    dict
        Collection of batched features required for running TERMinator. This contains:

        - :code:`msas` - the sequences for each TERM match to the target structure

        - :code:`features` - the :math:`\\phi, \\psi, \\omega`, and environment values of the TERM matches

        - :code:`ppoe` - the :math:`\\phi, \\psi, \\omega`, and environment values of the target structure

        - :code:`seq_lens` - lengths of the target sequences

        - :code:`focuses` - the corresponding target structure residue index for each TERM residue

        - :code:`contact_idxs` - contact indices for each TERM residue

        - :code:`src_key_mask` - mask for TERM residue padding

        - :code:`X` - coordinates

        - :code:`x_mask` - mask for the target structure

        - :code:`seqs` - the target sequences

        - :code:`ids` - the PDB ids

        - :code:`chain_idx` - the chain IDs
    """
    if self.msa_type:
        if self.msa_type.find("sample") > -1:
            for data in b_idx:
                seq_ids = np.array(data[0]['seq_id'])
                inds = np.argsort(seq_ids)
                seq_ids = seq_ids[inds[::-1]]
                cur_seqs = data[0]['sequence'][inds[::-1]]
                native_seq = cur_seqs[0]
                num_above_cutoff = np.sum(seq_ids >= 0.5)
                seq_ids = seq_ids[seq_ids > 0.5]
                cur_seqs = cur_seqs[:num_above_cutoff]
                if self.msa_type.find("cutoff") > -1:
                    cutoff = float(self.msa_type.split('_')[-1])
                    num_above_cutoff = np.sum(seq_ids >= cutoff)
                    seq_ids = seq_ids[seq_ids < cutoff]
                    cur_seqs = cur_seqs[num_above_cutoff:]
                num_possible_sequences = len(cur_seqs)
                if num_possible_sequences == 0:
                    data[0]['sequence'] = np.array(native_seq, dtype=int)
                else:
                    randomly_selected_sequence = random.randint(0, num_possible_sequences-1)
                    data[0]['sequence'] = np.array(cur_seqs[randomly_selected_sequence], dtype=int)
        elif self.msa_type.find("single") > -1:
            for data in b_idx:
                seq_ids = np.array(data[0]['seq_id'])
                inds = np.argsort(seq_ids)
                cur_seqs = data[0]['sequence'][inds[::-1]]
                data[0]['sequence'] = np.array(cur_seqs[0], dtype=int)
                seq_ids = seq_ids[inds[::-1]]
                data[0]['seq_id'] = np.array(seq_ids[0], dtype=float)
    return _wds_package(b_idx, self.use_sc, self.sc_mask_rate, self.base_sc_mask, self.chain_mask, self.sc_mask_schedule, self.sc_info, self.sc_mask, self.mask_neighbors, self.mask_interface, self.half_interface, self.inter_cutoff, self.sc_screen, self.sc_screen_range, self.sc_noise, self.epoch, self.msa_type, self.msa_id_cutoff, self.flex_type, self.replicate, self.noise_level, self.bond_length_noise_level, self.bond_angle_noise_level, self.noise_lim, self.pair_etab_dir, self.esm, self.batch_converter, self.use_esm_attns, self.use_reps, self.post_esm_mask, self.from_rla, self.esm_embed_layer, self.fix_seed, self.connect_chains, self.convert_to_esm, self.one_hot)

def WDSBatchSampler_len(self):
    """Returns length of dataset, i.e. number of batches.

    Returns
    -------
    int
        length of dataset.
    """
    if not self.ddp:
        return len(self.clusters)
    if len(self.clusters) % self.num_replicas != 0:
        if not self.drop_last:
            return math.floor((len(self.clusters) + self.num_replicas) / self.num_replicas)
        else:
            return math.floor((len(self.clusters)) / self.num_replicas)
    return math.ceil(len(self.clusters) / self.num_replicas)

def WDSBatchSampler_iter(self):
    """Allows iteration over dataset."""
    if not self.ddp:
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch
    else:
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.clusters), generator=g).tolist()
        else:
            indices = list(range(len(self.clusters)))
        
        self.num_samples = self.__len__()
        self.total_size = self.num_samples * self.num_replicas
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        for index in indices:
            yield self.clusters[index]


class WDSBatchSamplerWrapper(object):
    """BatchSampler/Dataloader helper class for TERM data using TERMDataset.

    Attributes in wrapped TERMBatchSampler class
    ----
    size: int
        Length of the dataset
    dataset: List
        List of features from TERM dataset
    total_term_lengths: List
        List of TERM lengths from the given dataset
    seq_lengths: List
        List of sequence lengths from the given dataset
    lengths: List
        TERM lengths or sequence lengths, depending on
        whether :code:`max_term_res` or :code:`max_seq_tokens` is set.
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_term_res : int or None, default=55000
        When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
    max_seq_tokens : int or None, default=None
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.
    """
    def __init__(self,
                 ddp):
        self.ddp = ddp
        if ddp:
            print('using distributed sampler!')
            class WDSBatchSampler(DistributedSampler):
                pass
        else:
            class WDSBatchSampler(Sampler):
                pass
        WDSBatchSampler.__init__ = WDSBatchSampler_init
        WDSBatchSampler._cluster = WDSBatchSampler_cluster
        WDSBatchSampler.package = WDSBatchSampler_package
        WDSBatchSampler.__len__ = WDSBatchSampler_len
        WDSBatchSampler.__iter__ = WDSBatchSampler_iter
        WDSBatchSampler._set_epoch = WDSBatchSampler_set_epoch
        self.sampler = WDSBatchSampler


# needs to be outside of object for pickling reasons (?)
def read_lens(in_folder, pdb_id, min_protein_len=30):
    """ Reads the lengths specified in the proper .length file and return them.

    If the read sequence length is less than :code:`min_protein_len`, instead return None.

    Args
    ----
    in_folder : str
        folder to find TERM file.
    pdb_id : str
        PDB ID to load.
    min_protein_len : int
        minimum cutoff for loading TERM file.
    Returns
    -------
    pdb_id : str
        PDB ID that was loaded
    total_term_length : int
        number of TERMS in file
    seq_len : int
        sequence length of file, or None if sequence length is less than :code:`min_protein_len`
    """
    path = f"{in_folder}/{pdb_id}/{pdb_id}.length"
    # pylint: disable=unspecified-encoding
    with open(path, 'rt') as fp:
        total_term_length = int(fp.readline().strip())
        seq_len = int(fp.readline().strip())
        if seq_len < min_protein_len:
            return None
    return pdb_id, total_term_length, seq_len


class TERMLazyDataset(Dataset):
    """TERM Dataset that loads all feature files into a Pytorch Dataset-like structure.

    Unlike TERMDataset, this just loads feature filenames, not actual features.

    Attributes
    ----
    dataset : list
        list of tuples containing feature filenames, TERM length, and sequence length
    shuffle_idx : list
        array of indices for the dataset, for shuffling
    """
    def __init__(self, in_folder, pdb_ids=None, min_protein_len=30, num_processes=32):
        """
        Initializes current TERM dataset by reading in feature files.

        Reads in all feature files from the given directory, using multiprocessing
        with the provided number of processes. Stores the feature filenames, the TERM length,
        and the sequence length as a tuple representing the data. Can read from PDB ids or
        file paths directly. Uses the given protein length as a cutoff.

        Args
        ----
        in_folder : str
            path to directory containing feature files generated by :code:`scripts/data/preprocessing/generateDataset.py`
        pdb_ids: list, optional
            list of pdbs from `in_folder` to include in the dataset
        min_protein_len: int, default=30
            minimum length of a protein in the dataset
        num_processes: int, default=32
            number of processes to use during dataloading
        """
        self.dataset = []

        with mp.Pool(num_processes) as pool:

            if pdb_ids:
                print("Loading feature file paths")
                progress = tqdm(total=len(pdb_ids))

                def update_progress(res):
                    del res
                    progress.update(1)

                res_list = [
                    pool.apply_async(read_lens, (in_folder, pdb_id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for pdb_id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        pdb_id, total_term_length, seq_len = data
                        filename = f"{in_folder}/{pdb_id}/{pdb_id}.features"
                        self.dataset.append((os.path.abspath(filename), total_term_length, seq_len))
            else:
                print("Loading feature file paths")

                filelist = list(glob.glob(f'{in_folder}/*/*.features'))
                progress = tqdm(total=len(filelist))

                def update_progress(res):
                    del res
                    progress.update(1)

                # get pdb_ids
                pdb_ids = [os.path.basename(path)[:-len(".features")] for path in filelist]

                res_list = [
                    pool.apply_async(read_lens, (in_folder, pdb_id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for pdb_id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        pdb_id, total_term_length, seq_len = data
                        filename = f"{in_folder}/{pdb_id}/{pdb_id}.features"
                        self.dataset.append((os.path.abspath(filename), total_term_length, seq_len))

        self.shuffle_idx = np.arange(len(self.dataset))

    def shuffle(self):
        """Shuffle the dataset"""
        np.random.shuffle(self.shuffle_idx)

    def __len__(self):
        """Returns length of the given dataset.

        Returns
        -------
        int
            length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Extract a given item with provided index.

        Args
        ----
        idx : int
            Index of item to return.
        Returns
        ----
        data : dict
            Data from TERM file (as dict)
        total_term_len : int
            Sum of lengths of all TERMs
        seq_len : int
            Length of protein sequence
        """
        data_idx = self.shuffle_idx[idx]
        if isinstance(data_idx, list):
            return [self.dataset[i] for i in data_idx]
        return self.dataset[data_idx]


class TERMLazyBatchSampler(Sampler):
    """BatchSampler/Dataloader helper class for TERM data using TERMLazyDataset.

    Attributes
    ----------
    dataset : TERMLazyDataset
        Dataset to batch.
    size : int
        Length of dataset
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_term_res : int or None, default=55000
        When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
    max_seq_tokens : int or None, default=None
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.
    term_matches_cutoff : int or None, default=None
        Use the top :code:`term_matches_cutoff` TERM matches for featurization.
        If :code:`None`, apply no cutoff.
    term_dropout : str or None, default=None
        Let `t` be the number of TERM matches in the given datapoint.
        Select a random int `n` from 1 to `t`, and take a random subset `n`
        of the given TERM matches to keep. If :code:`term_dropout='keep_first'`,
        keep the first match and choose `n-1` from the rest.
        If :code:`term_dropout='all'`, choose `n` matches from all matches.
    """
    def __init__(self,
                 dataset,
                 batch_size=4,
                 sort_data=False,
                 shuffle=True,
                 semi_shuffle=False,
                 semi_shuffle_cluster_size=500,
                 batch_shuffle=True,
                 drop_last=False,
                 max_term_res=55000,
                 max_seq_tokens=None,
                 term_matches_cutoff=None,
                 term_dropout=None):
        """
        Reads in and processes a given dataset.

        Given the provided dataset, load all the data. Then cluster the data using
        the provided method, either shuffled or sorted and then shuffled.

        Args
        ----
        dataset : TERMLazyDataset
            Dataset to batch.
        batch_size : int or None, default=4
            Size of batches created. If variable sized batches are desired, set to None.
        sort_data : bool, default=False
            Create deterministic batches by sorting the data according to the
            specified length metric and creating batches from the sorted data.
            Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
        shuffle : bool, default=True
            Shuffle the data completely before creating batches.
            Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
        semi_shuffle : bool, default=False
            Sort the data according to the specified length metric,
            then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
            Within each partition perform a complete shuffle. The upside is that
            batching with similar lengths reduces padding making for more efficient computation,
            but the downside is that it does a less complete shuffle.
        semi_shuffle_cluster_size : int, default=500
            Size of partition to use when :code:`semi_shuffle=True`.
        batch_shuffle : bool, default=True
            If set to :code:`True`, shuffle samples within a batch.
        drop_last : bool, default=False
            If set to :code:`True`, drop the last samples if they don't form a complete batch.
        max_term_res : int or None, default=55000
            When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
            batch by fitting as many datapoints as possible with the total number of
            TERM residues included below `max_term_res`.
            Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
        max_seq_tokens : int or None, default=None
            When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
            batch by fitting as many datapoints as possible with the total number of
            sequence residues included below `max_seq_tokens`.
        term_matches_cutoff : int or None, default=None
            Use the top :code:`term_matches_cutoff` TERM matches for featurization.
            If :code:`None`, apply no cutoff.
        term_dropout : str or None, default=None
            Let `t` be the number of TERM matches in the given datapoint.
            Select a random int `n` from 1 to `t`, and take a random subset `n`
            of the given TERM matches to keep. If :code:`term_dropout='keep_first'`,
            keep the first match and choose `n-1` from the rest.
            If :code:`term_dropout='all'`, choose `n` matches from all matches.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.size = len(dataset)
        self.filepaths, self.total_term_lengths, self.seq_lengths = zip(*dataset)
        assert not (max_term_res is None
                    and max_seq_tokens is None), "Exactly one of max_term_res and max_seq_tokens must be None"
        if max_term_res is None and max_seq_tokens > 0:
            self.lengths = self.seq_lengths
        elif max_term_res > 0 and max_seq_tokens is None:
            self.lengths = self.total_term_lengths
        else:
            raise Exception("Exactly one of max_term_res and max_seq_tokens must be None")
        self.shuffle = shuffle
        self.sort_data = sort_data
        self.batch_shuffle = batch_shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_term_res = max_term_res
        self.max_seq_tokens = max_seq_tokens
        self.semi_shuffle = semi_shuffle
        self.semi_shuffle_cluster_size = semi_shuffle_cluster_size
        self.term_matches_cutoff = term_matches_cutoff
        assert term_dropout in ["keep_first", "all", None], f"term_dropout={term_dropout} is not a valid argument"
        self.term_dropout = term_dropout

        assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make clusters of indices corresponding to batches of data.

        This method speeds up training by sorting data points with similar TERM lengths
        together, if :code:`sort_data` or :code:`semi_shuffle` are on. Under `sort_data`,
        the data is sorted by length. Under `semi_shuffle`, the data is broken up
        into clusters based on length and shuffled within the clusters. Otherwise,
        it is randomly shuffled. Data is then loaded into batches based on the number
        of proteins that will fit into the GPU without overloading it, based on
        :code:`max_term_res` or :code:`max_seq_tokens`.
        """

        # if we sort data, use sorted indexes instead
        if self.sort_data:
            idx_list = np.argsort(self.lengths)
        elif self.semi_shuffle:
            # trying to speed up training
            # by shuffling points with similar term res together
            idx_list = np.argsort(self.lengths)
            shuffle_borders = []

            # break up datapoints into large clusters
            border = 0
            while border < len(self.lengths):
                shuffle_borders.append(border)
                border += self.semi_shuffle_cluster_size

            # shuffle datapoints within clusters
            last_cluster_idx = len(shuffle_borders) - 1
            for cluster_idx in range(last_cluster_idx + 1):
                start = shuffle_borders[cluster_idx]
                if cluster_idx < last_cluster_idx:
                    end = shuffle_borders[cluster_idx + 1]
                    np.random.shuffle(idx_list[start:end])
                else:
                    np.random.shuffle(idx_list[start:])

        else:
            idx_list = list(range(len(self.dataset)))
            np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.batch_size is None:
            if self.max_term_res is None and self.max_seq_tokens > 0:
                cap_len = self.max_seq_tokens
            elif self.max_term_res > 0 and self.max_seq_tokens is None:
                cap_len = self.max_term_res

            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(idx_list):
                current_batch_lens.append(self.lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > cap_len:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.lengths[idx]]
                else:
                    batch.append(idx)

        else:  # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters

    def package(self, b_idx):
        """Package the given datapoints into tensors based on provided indices.

        Tensors are extracted from the data and padded. Coordinates are featurized
        and the length of TERMs and chain IDs are added to the data.

        Args
        ----
        b_idx : list of (str, int, int)
            The path to the feature files, the sum of the lengths of all TERMs, and the sum of all sequence lengths
            for each datapoint to package.

        Returns
        -------
        dict
            Collection of batched features required for running TERMinator. This contains:

            - :code:`msas` - the sequences for each TERM match to the target structure

            - :code:`features` - the :math:`\\phi, \\psi, \\omega`, and environment values of the TERM matches

            - :code:`ppoe` - the :math:`\\phi, \\psi, \\omega`, and environment values of the target structure

            - :code:`seq_lens` - lengths of the target sequences

            - :code:`focuses` - the corresponding target structure residue index for each TERM residue

            - :code:`contact_idxs` - contact indices for each TERM residue

            - :code:`src_key_mask` - mask for TERM residue padding

            - :code:`X` - coordinates

            - :code:`x_mask` - mask for the target structure

            - :code:`seqs` - the target sequences

            - :code:`ids` - the PDB ids

            - :code:`chain_idx` - the chain IDs
        """
        if self.batch_shuffle:
            b_idx_copy = b_idx[:]
            random.shuffle(b_idx_copy)
            b_idx = b_idx_copy

        # load the files specified by filepaths
        batch = []
        for data in b_idx:
            filepath = data[0]
            with open(filepath, 'rb') as fp:
                batch.append((pickle.load(fp), data[1]))
                if 'ppoe' not in batch[-1][0].keys():
                    print(filepath)

        # package batch
        packaged_batch = _package(batch)

        features = packaged_batch["features"]
        msas = packaged_batch["msas"]
        # apply TERM matches cutoff
        if self.term_matches_cutoff:
            features = features[:, :self.term_matches_cutoff]
            msas = msas[:, :self.term_matches_cutoff]
        # apply TERM matches dropout
        if self.term_dropout:
            # sample a random number of alignments to keep
            n_batch, n_align, n_terms, n_features = features.shape
            if self.term_dropout == 'keep_first':
                n_keep = torch.randint(0, n_align, [1]).item()
            elif self.term_dropout == 'all':
                n_keep = torch.randint(1, n_align, [1]).item()
            # sample from a multinomial distribution
            weights = torch.ones([1, 1]).expand([n_batch * n_terms, n_keep])
            if n_keep == 0:
                sample_idx = torch.ones(1)
            else:
                sample_idx = torch.multinomial(weights, n_keep)
                sample_idx = sample_idx.view([n_batch, n_terms, n_keep]).transpose(-1, -2)
                sample_idx_features = sample_idx.unsqueeze(-1).expand([n_batch, n_keep, n_terms, n_features])
                sample_idx_msas = sample_idx

            if self.term_dropout == 'keep_first':
                if n_keep == 0:
                    features = features[:, 0:1]
                    msas = msas[:, 0:1]
                else:
                    sample_features = torch.gather(features[:, 1:], 1, sample_idx_features)
                    sample_msas = torch.gather(msas[:, 1:], 1, sample_idx_msas)
                    features = torch.cat([features[:, 0:1], sample_features], dim=1)
                    msas = torch.cat([msas[:, 0:1], sample_msas], dim=1)
            elif self.term_dropout == 'all':
                features = torch.gather(features, 1, sample_idx_features)
                msas = torch.gather(msas, 1, sample_idx_msas)

        packaged_batch["features"] = features
        packaged_batch["msas"] = msas

        return packaged_batch

    def __len__(self):
        """Returns length of dataset, i.e. number of batches.

        Returns
        -------
        int
            length of dataset.
        """
        return len(self.clusters)

    def __iter__(self):
        """Allows iteration over dataset."""
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch