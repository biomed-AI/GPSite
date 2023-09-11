import string, os
import numpy as np

MAX_INPUT_SEQ = 1000
MAX_SEQ_LEN = 1500

nn_config = {
    'node_input_dim': 1024 + 9 + 184,
    'edge_input_dim': 450,
    'hidden_dim': 128,
    'layer': 4,
    'augment_eps': 0.1,
    'dropout': 0.2
}


# deal with IDs with different formats: e.g. "sp|P05067|A4_HUMAN Amyloid-beta precursor protein" (UniProt), "7PRW_1|Chains A, B|Glucocorticoid receptor|Homo sapiens" (PDB)
def get_ID(name):
    name = name.split("|")
    ID = "_".join(name[0:min(2, len(name))])
    ID = ID.replace(" ", "_")
    return ID


def remove_non_standard_aa(seq):
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    new_seq = ""
    for aa in seq:
        if aa in standard_aa:
            new_seq += aa
    return new_seq


def process_fasta(fasta_file, outpath):
    ID_list = []
    seq_list = []

    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            ID_list.append(get_ID(line[1:-1]))
        elif line[0] in string.ascii_letters:
            seq = line.strip().upper()
            seq = remove_non_standard_aa(seq)
            seq_list.append(seq[0:min(MAX_SEQ_LEN, len(seq))]) # trim long sequence

    if len(ID_list) == len(seq_list):
        if len(ID_list) > MAX_INPUT_SEQ:
            return 1
        else:
            new_fasta = "" # with processed IDs and seqs
            for i in range(len(ID_list)):
                new_fasta += (">" + ID_list[i] + "\n" + seq_list[i] + "\n")
            with open(outpath + "test_seq.fa", "w") as f:
                f.write(new_fasta)

            return [ID_list, seq_list]
    else:
        return -1


def export_predictions(predictions, seq_list, outpath):
    # original order: ["PRO", "PEP", "DNA", "RNA", "ZN", "CA", "MG", "MN", "ATP", "HEME"]
    thresholds = [0.35, 0.47, 0.41, 0.46, 0.73, 0.57, 0.44, 0.65, 0.51, 0.61] # select by maximizing MCC on the cross validation
    index = [2, 3, 1, 0, 8, 9, 4, 5, 6, 7] # switch order to ["DNA", "RNA", "PEP", "PRO", "ATP", "HEME", "ZN", "CA", "MG", "MN"]
    GPSite_binding_scores = {}

    for i, ID in enumerate(predictions):
        seq = seq_list[i]
        preds = predictions[ID]
        norm_preds = []
        binding_scores = [] # protein-level binding scores

        for lig_idx, pred in enumerate(preds):
            threshold = thresholds[lig_idx]

            norm_pred = []
            for score in pred:
                if score > threshold:
                    norm_score = (score - threshold) / (1 - threshold) * 0.5 + 0.5
                else:
                    norm_score = (score / threshold) * 0.5
                norm_pred.append(norm_score)
            norm_preds.append(norm_pred)

            if lig_idx in [4, 5, 6, 7]: # metal ions
                k = 5
            else:
                k = 10
            k = min(k, len(seq))

            idx = np.argpartition(norm_pred, -k)[-k:]
            topk_norm_sores = np.array(norm_pred)[idx]
            binding_scores.append(topk_norm_sores.mean())

        GPSite_binding_scores[ID] = binding_scores


        pred_txt = "No.\tAA\tDNA_binding\tRNA_binding\tPeptide_binding\tProtein_binding\tATP_binding\tHEM_binding\tZN_binding\tCA_binding\tMG_binding\tMN_binding\n"
        for j in range(len(seq)):
            pred_txt += "{}\t{}".format(j+1, seq[j]) # 1-based

            for idx in index:
                norm_score = norm_preds[idx][j]
                pred_txt += "\t{:.3f}".format(norm_score)

            pred_txt += "\n"

        with open("{}/pred/{}.txt".format(outpath, ID), "w") as f:
            f.write(pred_txt)


        # export the predictions to a pdb file (for the visualization in the server)
        '''
        score_lines = []
        for j in range(len(seq)):
            score_line = ""
            for idx in index:
                score = norm_preds[idx][j]
                score = "{:.2f}".format(score * 100)
                score = " " * (6 - len(score)) + score
                score_line += score
            score_lines.append(score_line)

        with open("{}/pdb/{}.pdb".format(outpath, ID), "r") as f:
            lines = f.readlines()

        current_pos = -1
        new_pdb = ""
        for line in lines:
            if line[0:4] != "ATOM":
                continue
            if int(line[22:26].strip()) != current_pos:
                current_pos = int(line[22:26].strip())
                score_line = score_lines.pop(0)
            new_line = line[0:60] + score_line + "           " + line.strip()[-1] + "  \n"
            new_pdb += new_line
        new_pdb += "TER\n"

        with open("{}/pred/{}.pdb".format(outpath, ID), "w") as f:
            f.write(new_pdb)
        '''


    with open(outpath + "esmfold_pred.log", "r") as f:
        lines = f.readlines()

    info_dict = {}
    for line in lines:
        if "pLDDT" in line:
            ID_len, pLDDT, pTM = line.strip().split("|")[-1].strip().split(",")[0:3]
            ID = ID_len.strip().split()[3]
            length = ID_len.strip().split()[6]
            pLDDT = float(pLDDT.strip().split()[1])
            pTM = float(pTM.strip().split()[1])
            info_dict[ID] = [length, pLDDT, pTM]

    entry_info = "ID\tLength\tpLDDT\tpTM\tDNA_Binding\tRNA_Binding\tPeptide_Binding\tProtein_Binding\tATP_Binding\tHEM_Binding\tZN_Binding\tCA_Binding\tMG_Binding\tMN_Binding\n"
    for ID in predictions:
        Length, pLDDT, pTM = info_dict[ID]
        binding_scores = GPSite_binding_scores[ID]
        binding_scores = np.array(binding_scores)[index] # switch order to DNA, RNA, PEP ...
        entry = "{}\t{}\t{}\t{:.3f}".format(ID, Length, pLDDT, pTM)
        for score in binding_scores:
            entry += "\t{:.3f}".format(score)
        entry_info += (entry + "\n")

    with open("{}/pred/overview.txt".format(outpath), "w") as f:
        f.write(entry_info)

    os.system("rm {}/esmfold_pred.log".format(outpath))
