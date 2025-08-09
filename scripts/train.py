import os
import json
import sys

import japanize_matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).resolve().parent.parent)))
from utils.tokenizers import SPETokenizerWrapper
from modules import models
from modules.models import SmilesPredictor3dimFreqIrRaman
from modules.trainer import SmilesTrainer3dimFreqIrRaman
from constants import (
    device,
    SCRIPT_OVER_WRITE,
    SMILES_MAX_LENGTH,
    SMILES_VOCAB_SIZE,
    SPECTRUM_MAX_LENGTH,
    DROPOUT_RATE,
    LR,
    MODEL_NAME,
    ANNEALING_START_STEP,
    LR_PLATEAU_FACTOR,
    ENCODER_NUM_LAYERS,
    ENCODER_HIDDEN_DIMENSION,
    ENCODER_N_HEADS,
    DECODER_NUM_LAYERS,
    DECODER_N_HEADS,
    DECODER_HIDDEN_DIMENSION,
)
tokenizer = SPETokenizerWrapper()


def main(BATCH_SIZE):

    model_params = {
        "encoder_num_layers":ENCODER_NUM_LAYERS,
        "smiles_max_length": SMILES_MAX_LENGTH,
        "spectrum_max_length": SPECTRUM_MAX_LENGTH,
        "encoder_hidden_dimention": ENCODER_HIDDEN_DIMENSION,
        "encoder_n_heads": ENCODER_N_HEADS,
        "encoder_dropout_rate": DROPOUT_RATE,
        "decoder_hidden_dimention": DECODER_HIDDEN_DIMENSION,
        "decoder_n_heads": DECODER_N_HEADS,
        "decoder_dropout_rate": DROPOUT_RATE,
        "decoder_num_layers": DECODER_NUM_LAYERS,
        "smiles_vocab_size": SMILES_VOCAB_SIZE, #include #[EOS], [BOS], " "
        "embed_dimention": EMBEDDING_DIMENTION,
        "embed_dropout_rate": DROPOUT_RATE,
    }

    SAVE_DIRECTORY_NAME = f"{MODEL_NAME}"
    BASE_DIRECTORY =  str((Path(__file__).resolve().parent.parent)) + "Result_directory/"
    MODEL_DIRECTORY = "Trained_models"
    RESULT_DIRECTOROY = "Result_graphs"
    os.makedirs(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/", exist_ok=True)
    os.makedirs(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/scripts", exist_ok=True)
    os.makedirs(f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/", exist_ok=True)

    training_params = {
        "init_lr": LR,
        "wamup":False,
        "warmup_end_step":0,
        "warmup_start_factor":1 * (10 ** -2),
        "warmup_end_factor":1,
        # "lr_start_factor":1,
        # "lr_end_factor":LR_ANNEALING_STOP_FACTOR ,
        "lr_annealing_start_step":ANNEALING_START_STEP,
        # "lr_annealing_total_steps":ANNEALING_STOP_STEP,
        "lr_plateau_factor":LR_PLATEAU_FACTOR,
        "lr_plateau_patience":10,
        "lr_plateau_threshold":1e-4,
        "model_check_interval": 1,
        "clip_max_grad_norm": 1,
        "validation_step_interval_rate": 1,
        "save_model_name": f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/model.pt",
        "loss_fig_name": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/loss.png",
        "small_train_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_train_loss.txt",
        "small_valid_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_valid_loss.txt",
        "train_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/train_loss.txt",
        "valid_loss_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/valid_loss.txt",
        "small_valid_accuracy_filename": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_valid_accuracy.txt",
        "small_accuracy_fig_name": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/small_accuracy.png",
        "big_accuracy_fig_name": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/big_accuracy.png",
        "title": f"Functional_{MODEL_NAME}",
        "num_epochs": 10000,
        "patience": 1000000000000, #inf
        "small_val_ratio": 0.01,
        "script_save_dirctory": f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/scripts",
        "train_script_path": __file__,
        "module_directory_path": "/home/Futo/IR_and_Raman/modules",
        "bos_indice": int(tokenizer.VOCABS_INDICES["[BOS]"]),
        "num_label": 1,
        "script_over_write":SCRIPT_OVER_WRITE,
        "lr_record_path": f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/lr_record.txt",
    }
    with open(training_params["lr_record_path"], "w") as f:
        pass

    print(int(tokenizer.VOCABS_INDICES["[BOS]"]))
    """
    データセット
    """
    from torch.utils.data import DataLoader, TensorDataset
    src_train_freq = torch.load(os.path.join(DATA_DIRECTORY, "train/freqs.pt"), map_location="cpu")
    src_val_freq = torch.load(os.path.join(DATA_DIRECTORY, "valid/freqs.pt"), map_location="cpu")
    src_test_freq = torch.load(os.path.join(DATA_DIRECTORY, "test/freqs.pt"), map_location="cpu")
    src_train_IR = torch.load(os.path.join(DATA_DIRECTORY, "train/IRs.pt"), map_location="cpu")
    src_val_IR = torch.load(os.path.join(DATA_DIRECTORY, "valid/IRs.pt"), map_location="cpu")
    src_test_IR = torch.load(os.path.join(DATA_DIRECTORY, "test/IRs.pt"), map_location="cpu")
    src_train_Raman = torch.load(os.path.join(DATA_DIRECTORY, "train/Ramans.pt"), map_location="cpu")
    src_val_Raman = torch.load(os.path.join(DATA_DIRECTORY, "valid/Ramans.pt"), map_location="cpu")
    src_test_Raman = torch.load(os.path.join(DATA_DIRECTORY, "test/Ramans.pt"), map_location="cpu")
    src_train_attention_masks = torch.load(os.path.join(DATA_DIRECTORY, "train/freq_attention_masks.pt"), map_location="cpu")
    src_val_attention_masks = torch.load(os.path.join(DATA_DIRECTORY, "valid/freq_attention_masks.pt"), map_location="cpu")
    src_test_attention_masks = torch.load(os.path.join(DATA_DIRECTORY, "test/freq_attention_masks.pt"), map_location="cpu")
    composition_train_chars = torch.load(os.path.join(DATA_DIRECTORY, "train/compositions_char_tokenized.pt"), map_location="cpu")
    composition_valid_chars = torch.load(os.path.join(DATA_DIRECTORY, "valid/compositions_char_tokenized.pt"), map_location="cpu")
    composition_train_masks = torch.load(os.path.join(DATA_DIRECTORY, "train/compositions_char_attention_mask.pt"), map_location="cpu")
    composition_valid_masks = torch.load(os.path.join(DATA_DIRECTORY, "valid/compositions_char_attention_mask.pt"), map_location="cpu")
    # src_train_sparse_float_IR = torch.load(os.path.join(DATA_DIRECTORY, "train/sparse_float_IRs.pt"), map_location="cpu")
    # src_valid_sparse_float_IR = torch.load(os.path.join(DATA_DIRECTORY, "valid/sparse_float_IRs.pt"), map_location="cpu")
    # src_train_sparse_float_Raman = torch.load(os.path.join(DATA_DIRECTORY, "train/sparse_float_Ramans.pt"), map_location="cpu")
    # src_valid_sparse_float_Raman = torch.load(os.path.join(DATA_DIRECTORY, "valid/sparse_float_Ramans.pt"), map_location="cpu")
    # src_train_sparse_float_IR = src_train_sparse_float_IR.to(torch.float32)
    # src_valid_sparse_float_IR = src_valid_sparse_float_IR.to(torch.float32)
    # src_train_sparse_float_Raman = src_train_sparse_float_Raman.to(torch.float32)
    # src_valid_sparse_float_Raman = src_valid_sparse_float_Raman.to(torch.float32)

    #standardization
    # IR_min = min(src_train_IR.min(), src_val_IR.min(), src_test_IR.min())
    # IR_max = max(src_train_IR.max(), src_val_IR.max(), src_test_IR.max())
    # Raman_min = min(src_train_Raman.min(), src_val_Raman.min(), src_test_Raman.min())
    # Raman_max = max(src_train_Raman.max(), src_val_Raman.max(), src_test_Raman.max())

    # src_train_IR_standardized = (src_train_IR - IR_min) / (IR_max - IR_min)
    # src_val_IR_standardized = (src_val_IR - IR_min) / (IR_max - IR_min)
    # src_test_IR_standardized = (src_test_IR - IR_min) / (IR_max - IR_min)
    # src_train_Raman_standardized = (src_train_Raman - Raman_min) / (Raman_max - Raman_min)
    # src_val_Raman_standardized = (src_val_Raman - Raman_min) / (Raman_max - Raman_min)
    # src_test_Raman_standardized = (src_test_Raman - Raman_min) / (Raman_max - Raman_min)




    smiles_train = torch.load(os.path.join(DATA_DIRECTORY, "train/smiles_ids.pt"), map_location="cpu")
    smiles_valid = torch.load(os.path.join(DATA_DIRECTORY, "valid/smiles_ids.pt"), map_location="cpu")
    smiles_test = torch.load(os.path.join(DATA_DIRECTORY, "test/smiles_ids.pt"), map_location="cpu")
    smiles_attention_mask_train = torch.load(os.path.join(DATA_DIRECTORY, "train/smiles_attention_masks.pt"), map_location="cpu")
    smiles_attention_mask_valid = torch.load(os.path.join(DATA_DIRECTORY, "valid/smiles_attention_masks.pt"), map_location="cpu")
    smiles_attention_mask_test = torch.load(os.path.join(DATA_DIRECTORY, "test/smiles_attention_masks.pt"), map_location="cpu")

    # print("src_train_freq")
    # print(src_train_freq.shape)
    # print("src_train_IR")
    # print(src_train_IR.shape)
    # print("src_train_Raman")
    # print(src_train_Raman.shape)
    # print("src_train_attention_masks")
    # print(src_train_attention_masks.shape)
    # print("smiles_train")
    # print(smiles_train.shape)
    # print("smiles_attention_mask_train")
    # print(smiles_attention_mask_train.shape)

    # dataset_train = TensorDataset(src_train_freq, src_train_IR, src_train_Raman, src_train_attention_masks, smiles_train, smiles_attention_mask_train, composition_train_chars, composition_train_masks)
    # dataset_train = TensorDataset(src_train_freq[:DATA_POINT], src_train_IR[:DATA_POINT], src_train_Raman[:DATA_POINT], src_train_attention_masks[:DATA_POINT], smiles_train[:DATA_POINT], smiles_attention_mask_train[:DATA_POINT], composition_train_chars[:DATA_POINT], composition_train_masks[:DATA_POINT])
    # dataset_train = TensorDataset(src_train_sparse_float_IR[:DATA_POINT], composition_train_chars[:DATA_POINT], composition_train_masks[:DATA_POINT], smiles_train[:DATA_POINT], smiles_attention_mask_train[:DATA_POINT])
    dataset_train = TensorDataset(src_train_freq, src_train_IR, src_train_Raman, src_train_attention_masks, smiles_train, smiles_attention_mask_train)
    dataset_valid = TensorDataset(src_val_freq, src_val_IR, src_val_Raman, src_val_attention_masks, smiles_valid, smiles_attention_mask_valid)
    # dataset_train_for_eval = TensorDataset(tgt_train_for_eval, tgt_train_mask_for_eval)
    # dataset_test = TensorDataset(tgt_test, tgt_test_mask)
    train_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False)
    # train_dataloader_for_eval = DataLoader(dataset_train_for_eval, batch_size=BATCH_SIZE, shuffle=False)
    # test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
    # model = FunctionalPredictorFreqIrRaman(model_params)
    
    model = SmilesPredictor3dimFreqIrRaman(model_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_params["init_lr"]) #今はとりあえずwarmupを0に
    model = model.to(device)
    # optimizer = TransOptimizerWrapper(optimizer)
    trainer = SmilesTrainer3dimFreqIrRaman()

    #json化と保存
    import json
    with open(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/model_params.json", "w") as f:
        json.dump(model_params, f, indent=4)
    with open(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/training_params.json", "w") as f:
        json.dump(training_params, f, indent=4)


    model, train_loss_list, valid_loss_list = trainer.train_model_loop(
        model,
        {**training_params, "tokenizer_obj": tokenizer, "label_list":["reconstruct_rate"]},
        train_dataloader,
        valid_dataloader,
        optimizer,
        device,
    )

if __name__ == "__main__":
    main(BATCH_SIZE=4096)