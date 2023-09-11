import numpy as np
from tqdm import tqdm
import os, argparse, datetime

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from feature_extraction.ProtTrans import get_ProtTrans
from feature_extraction.process_structure import get_pdb_xyz, process_dssp, match_dssp
from utils import *
from model import *


############ Set to your own path! ############
ProtTrans_path = "/data/user/yuanqm/tools/Prot-T5-XL-U50"
###############################################

script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
model_path = os.path.dirname(script_path[0:-1]) + "/model/"


def extract_feat(ID_list, seq_list, outpath, gpu):
    max_len = max([len(seq) for seq in seq_list])
    chunk_size = 32 if max_len > 1000 else 64

    esmfold_cmd = "python {}/feature_extraction/esmfold.py -i {} -o {} --chunk-size {}".format(script_path, outpath + "test_seq.fa", outpath + "pdb/", chunk_size)
    if not gpu: # slow!!
        esmfold_cmd += " --cpu-only"
    else:
        esmfold_cmd = "CUDA_VISIBLE_DEVICES=" + gpu + " " + esmfold_cmd
    os.system(esmfold_cmd + " | tee {}/esmfold_pred.log".format(outpath))

    Min_protrans = torch.tensor(np.load(script_path + "feature_extraction/Min_ProtTrans_repr.npy"), dtype = torch.float32)
    Max_protrans = torch.tensor(np.load(script_path + "feature_extraction/Max_ProtTrans_repr.npy"), dtype = torch.float32)
    get_ProtTrans(ID_list, seq_list, Min_protrans, Max_protrans, ProtTrans_path, outpath, gpu)

    print("Processing PDB files...")
    for ID in tqdm(ID_list):
        with open(outpath + "pdb/" + ID + ".pdb", "r") as f:
            X = get_pdb_xyz(f.readlines()) # [L, 5, 3]
        torch.save(torch.tensor(X, dtype = torch.float32), outpath + "pdb/" + ID + '.tensor')

    print("Extracting DSSP features...")
    for i in tqdm(range(len(ID_list))):
        ID = ID_list[i]
        seq = seq_list[i]

        os.system("{}/feature_extraction/mkdssp -i {}/pdb/{}.pdb -o {}/DSSP/{}.dssp".format(script_path, outpath, ID, outpath, ID))
        dssp_seq, dssp_matrix = process_dssp("{}/DSSP/{}.dssp".format(outpath, ID))
        if dssp_seq != seq:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)

        torch.save(torch.tensor(np.array(dssp_matrix), dtype = torch.float32), "{}/DSSP/{}.tensor".format(outpath, ID))
        os.system("rm {}/DSSP/{}.dssp".format(outpath, ID))


def predict(ID_list, outpath, batch, gpu):
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')

    node_input_dim = nn_config['node_input_dim']
    edge_input_dim = nn_config['edge_input_dim']
    hidden_dim = nn_config['hidden_dim']
    layer = nn_config['layer']
    augment_eps = nn_config['augment_eps']
    dropout = nn_config['dropout']

    task_list = ["PRO", "PEP", "DNA", "RNA", "ZN", "CA", "MG", "MN", "ATP", "HEME"]

    # Test
    test_dataset = ProteinGraphDataset(ID_list, outpath)
    test_dataloader = DataLoader(test_dataset, batch_size = batch, shuffle=False, drop_last=False, num_workers=8, prefetch_factor=2)

    models = []
    for fold in range(5):
        state_dict = torch.load(model_path + 'fold%s.ckpt'%fold, device)
        model = GPSite(node_input_dim, edge_input_dim, hidden_dim, layer, augment_eps, dropout, task_list).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    test_pred_dict = {}
    for data in tqdm(test_dataloader):
        data = data.to(device)

        with torch.no_grad():
            outputs = [model(data.X, data.node_feat, data.edge_index, data.batch).sigmoid() for model in models]
            outputs = torch.stack(outputs,0).mean(0) # average the predictions from 5 models

        IDs = data.name
        outputs_split = torch_geometric.utils.unbatch(outputs, data.batch)
        for i, ID in enumerate(IDs):
            test_pred_dict[ID] = []
            for j in range(len(task_list)):
                test_pred_dict[ID].append(list(outputs_split[i][:,j].detach().cpu().numpy()))

    return test_pred_dict


def main(seq_info, outpath, batch, gpu):
    ID_list, seq_list = seq_info
    for dir_name in ["pdb", "ProtTrans", "DSSP", "pred"]:
        os.makedirs(outpath + dir_name, exist_ok = True)

    print("\n######## Feature extraction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    extract_feat(ID_list, seq_list, outpath, gpu)

    print("\n######## Feature extraction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    print("\n######## Prediction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    predictions = predict(ID_list, outpath, batch, gpu)

    print("\n######## Prediction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    export_predictions(predictions, seq_list, outpath)

    print("\n######## Results are saved in {} ########\n".format(outpath + "pred/"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--fasta", type = str, help = "Input fasta file", required=True)
    parser.add_argument("-o", "--outpath", type = str, help = "Output path to save intermediate files and final predictions", required=True)
    parser.add_argument("-b", "--batch", type = int, default = 4, help = "Batch size for GPSite prediction")
    parser.add_argument("--gpu", type = str, default = None, help = "The GPU id used for feature extraction and binding site prediction")

    args = parser.parse_args()
    run_id = args.fasta.split("/")[-1].split(".")[0].replace(" ", "_")
    outpath = args.outpath + "/" + run_id + "/"
    os.makedirs(outpath, exist_ok = True)

    seq_info = process_fasta(args.fasta, outpath)

    if seq_info == -1:
        print("The format of your input fasta file is incorrect! Please check!")
    elif seq_info == 1:
        print("Too much sequences! Up to {} sequences are supported each time!".format(MAX_INPUT_SEQ))
    else:
        main(seq_info, outpath, args.batch, args.gpu)
