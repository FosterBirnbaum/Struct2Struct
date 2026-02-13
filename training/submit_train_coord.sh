#!/bin/bash

#SBATCH --mincpu=20
#SBATCH --gres=gpu:volta:1
#SBATCH --time=96:00:00
#SBATCH -o /data1/groups/keatinglab/fosterb/ProteinMPNN_experiments/coord_data/train-output_run.out
#SBATCH -e /data1/groups/keatinglab/fosterb/ProteinMPNN_experiments/coord_data/train-error_run.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2022b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator_fine_tune

python /data1/groups/keatinglab/ProteinMPNN/ProteinMPNN/training/coord_training.py \
           --path_for_outputs "/data1/groups/keatinglab/fosterb/ProteinMPNN_experiments/coord_data/" \
           --path_for_training_data "/data1/groups/keatinglab/fosterb/ingraham_data" \
           --num_examples_per_epoch 500000 \
           --save_model_every_n_epochs 5 \
		   --max_protein_length 8000 \
		   --batch_size 8000 \
		   --model_hparams "/data1/groups/keatinglab/ProteinMPNN/ProteinMPNN/training/coord_mpnn_model.json" \
		   --run_hparams "/data1/groups/keatinglab/ProteinMPNN/ProteinMPNN/training/coord_mpnn_run.json"
