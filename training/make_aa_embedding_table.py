import argparse
import os

import numpy as np
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX-"


def residue_vector_from_output(embeddings: torch.Tensor) -> torch.Tensor:
    """Extract a single-residue embedding from ESM-C output embeddings."""
    if embeddings.dim() != 3:
        raise ValueError(f"Expected embeddings [B, L, D], got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {embeddings.shape[0]}")

    # Typical tokenization yields <cls> A <eos> for one residue.
    if embeddings.shape[1] >= 3:
        return embeddings[0, 1, :]
    return embeddings[0, 0, :]


def build_table(model_name: str, device: str, vocab: int) -> torch.Tensor:
    tokens = ALPHABET[:vocab]
    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for aa in tokens:
            protein = ESMProtein(sequence=aa)
            protein_tensor = model.encode(protein)
            out = model.logits(protein_tensor, LogitsConfig(return_embeddings=True))
            if out.embeddings is None:
                raise RuntimeError("ESM-C logits call did not return embeddings")
            rows.append(residue_vector_from_output(out.embeddings).float().cpu())

    table = torch.stack(rows, dim=0)
    return table


def save_table(table: torch.Tensor, out_path: str) -> None:
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".npy":
        np.save(out_path, table.numpy())
    elif ext == ".npz":
        np.savez(out_path, aa_embedding_table=table.numpy())
    elif ext in (".pt", ".pth"):
        torch.save(table, out_path)
    else:
        raise ValueError("Unsupported output extension. Use .npy, .npz, .pt, or .pth")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aa_embedding_table for ESM-C contrastive training")
    parser.add_argument("--model", type=str, default="esmc_300m", help="ESM-C model name for ESMC.from_pretrained")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocab", type=int, default=21, choices=[21, 22], help="Token vocabulary size used by training")
    parser.add_argument("--out", type=str, required=True, help="Output path (.npy/.npz/.pt/.pth)")
    args = parser.parse_args()

    table = build_table(args.model, args.device, args.vocab)
    print(f"Built table with shape {tuple(table.shape)}")
    save_table(table, args.out)
    print(f"Saved aa_embedding_table to {args.out}")


if __name__ == "__main__":
    main()
