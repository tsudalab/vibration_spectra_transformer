import os
import json
import sys

import japanize_matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
home = os.path.expanduser("~")
sys.path.append(os.path.join(home, "IR_and_Raman"))
from utils.tokenizers import SPETokenizerWrapper
from modules import models
from modules.models import SmilesPredictor3dimFreqIrRaman
from modules.trainer import SmilesTrainer3dimFreqIrRaman
import time
tokenizer = SPETokenizerWrapper()

def main(BATCH_SIZE, MODEL_NUMBER, N, atom_size, device):
    LR = 1e-4
    # DATA_DIRECTORY = f"/home/Futo/IR_and_Raman/data_directory/GDB9/word_level_tokenized"
    DATA_DIRECTORY = f"/home/Futo/IR_and_Raman/data_directory/MMP05percent_extractLikeGDB/word_level_tokenized_{atom_size}"

    SAVE_DIRECTORY_NAME = f"{MODEL_NUMBER}"
    BASE_DIRECTORY = f"/home/Futo/IR_and_Raman/Result_directory/Smiles_predict_3dim_ir_raman_concatenate"
    MODEL_DIRECTORY = "Trained_models"
    RESULT_DIRECTOROY = "Result_graphs"
    os.makedirs(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/", exist_ok=True)
    os.makedirs(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/scripts", exist_ok=True)
    os.makedirs(f"{BASE_DIRECTORY}/{RESULT_DIRECTOROY}/{SAVE_DIRECTORY_NAME}/", exist_ok=True)
    """
    データセット
    """
    from torch.utils.data import DataLoader, TensorDataset
    # src_test_freq = torch.load(os.path.join(DATA_DIRECTORY, "test/freqs.pt"), map_location="cpu")
    # src_test_IR = torch.load(os.path.join(DATA_DIRECTORY, "test/IRs.pt"), map_location="cpu")
    # src_test_Raman = torch.load(os.path.join(DATA_DIRECTORY, "test/Ramans.pt"), map_location="cpu")
    # src_test_attention_masks = torch.load(os.path.join(DATA_DIRECTORY, "test/freq_attention_masks.pt"), map_location="cpu")
    # smiles_test = torch.load(os.path.join(DATA_DIRECTORY, "test/smiles_ids.pt"), map_location="cpu")
    # smiles_attention_mask_test = torch.load(os.path.join(DATA_DIRECTORY, "test/smiles_attention_masks.pt"), map_location="cpu")

    src_test_freq = torch.load(os.path.join(DATA_DIRECTORY, "freqs.pt"), map_location="cpu")
    src_test_IR = torch.load(os.path.join(DATA_DIRECTORY, "IRs.pt"), map_location="cpu")
    src_test_Raman = torch.load(os.path.join(DATA_DIRECTORY, "Ramans.pt"), map_location="cpu")
    src_test_attention_masks = torch.load(os.path.join(DATA_DIRECTORY, "freq_attention_masks.pt"), map_location="cpu")
    smiles_test = torch.load(os.path.join(DATA_DIRECTORY, "smiles_ids.pt"), map_location="cpu")
    smiles_attention_mask_test = torch.load(os.path.join(DATA_DIRECTORY, "smiles_attention_masks.pt"), map_location="cpu")


    dataset_test = TensorDataset(src_test_freq, src_test_IR, src_test_Raman, src_test_attention_masks, smiles_test, smiles_attention_mask_test)
    model: SmilesPredictor3dimFreqIrRaman = torch.load(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/model.pt")

    model = model.to(device)
    model.eval()
    trainer = SmilesTrainer3dimFreqIrRaman()
    with open(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/training_params.json") as f:
        training_params = json.load(f)
    training_params["tokenizer_obj"] = tokenizer
    training_params["smiles_max_length"] = 32
    start = time.time()
    topN_acc = trainer.eval_topN(model, training_params, dataset_test, N, device)
    topN_acc, average_CONF_num = trainer.eval_topN_with_CONFnum(model, training_params, dataset_test, N, device)
    end = time.time()
    print()
    print("BATCH_SIZE")
    print(BATCH_SIZE)
    print("topN_acc")
    print(topN_acc)
    print("time")
    print(end - start)

    # 保存
    with open(f"{BASE_DIRECTORY}/{MODEL_DIRECTORY}/{SAVE_DIRECTORY_NAME}/evaluation_atomsize{atom_size}_result_filterLikeGDB_result_withAverageCONFnum.txt_{BATCH_SIZE}_top{N}_update", "w") as f:
        f.write("topN_acc\n")
        f.write(str(topN_acc))
        f.write("\n")
        print("average_CONF_num\n")
        print(average_CONF_num)
        f.write("time\n")
        f.write(str(end - start))



    # model, train_loss_list, valid_loss_list = trainer.train_model_loop(
    #     model,
    #     {**training_params, "tokenizer_obj": tokenizer, "label_list":["reconstruct_rate"]}, #loop関数を統一するために仕方なく
    #     train_dataloader,
    #     valid_dataloader,
    #     optimizer,
    #     device,
    # )

if __name__ == "__main__":
    for model_number in ["Drop0_LR4_1", "Drop0_LR4_2", "Drop0_LR4_3"]:
        for N in [3, 5, 10, 20, 30, 40, 50, 100]:
            for atom_size in range(10, 12):
                main(BATCH_SIZE=4096, MODEL_NUMBER = model_number, N=N, atom_size=atom_size, device = "cuda:0")
    # main(BATCH_SIZE=4096, MODEL_NUMBER = "4_LR_change_3_1", N=5, device = "cuda:0")