#!/bin/bash

for i in {4..9}
do
  export HEAVY_SIZE=$i
  # python 4_1_freq_and_ir_and_raman_ready.py
  # python 4_1_functional_group_ready.py
  # python 4_1_smiles_ready.py
  # python 5_create_freq_attention_mask.py 
  # python 7_add_composition.py
  # python 10_composition_tokenize.py
  python 11_sparse_each.py
done