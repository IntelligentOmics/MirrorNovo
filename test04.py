import re
import config
import numpy as np
import csv

def readrestxt(path):
    res_dict = {}
    with open(path, "r")as f:
        data = f.read().split("\n")[1:]
    for line in data:
        if line:
            infolist = line.split("\t")
            res_dict["@".join(infolist[:2])] = infolist
    return res_dict

def readpNovoM(result_path):
    with open(result_path, "r")as f:
        buffer_result = f.read().split("\n")
    result_dict = {}
    for line in buffer_result:
        if line:
            if line.startswith("======") and "@" not in line:
            # if line.startswith("======") and not re.search("@", line):
                # title = line.split("\t")[0].split("@")[0][6:]
                # result_dict[title] = []
                pass
            elif line.startswith("======") and "@" in line:
                # title = line.split("\t")[0][6:]
                title = line.split("\t")[0][6:]
                result_dict[title] = []
            elif line.startswith("K") or line.startswith("R"):
                sequence = line.split("\t")[0]
                result_dict[title].append(sequence[1:])
            else:
                pass
        else:
            pass
    return result_dict

def read_mirror_spectra(path):
    with open(path, "r")as f:
        data = f.read().split("\n")[1:]
    mirror_list = []
    for line in data:
        if line:
            infolist = line.split("\t")
            mirror_list.append(infolist)
    return mirror_list

def readGCNovo(path):
    with open(path, "r")as f:
        data = f.read().split("\n")
    res_dict = {}
    for line in data:
        if line:
            if line.startswith("U"):
                title = line.split("\t")[0]
                res_dict[title] = []
            elif line.startswith("index"):
                pass
            elif line.startswith("BEGIN") or line.startswith("END"):
                pass
            else:
                psminfolist = line.split("\t")
                res_dict[title].append(psminfolist)

    return res_dict

def _match_AA_novor(target, predicted):
    """"""
    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    # print(target)
    target_mass = [config.mass_AA[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    # print(predicted)
    predicted_mass = [config.mass_AA[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)

    i = 0
    j = 0
    while i < target_len and j < predicted_len:
        if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
            if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                # ~ if  decoder_input[index_aa] == output[index_aa]:
                num_match += 1
            i += 1
            j += 1
        elif target_mass_cum[i] < predicted_mass_cum[j]:
            i += 1
        else:
            j += 1

    return num_match

def readpFindres(path):
    with open(path, "r")as f:
        data = f.read().split("\n")[1:]
    mirror_list = {}
    for line in data:
        if line:
            infolist = line.split("\t")
            mirror_list[infolist[0]] = infolist
    return mirror_list

def readfeaturecsv(path):
    feature_dict = {}
    with open(path, "r")as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader:
            infolist = list(line)
            feature_dict[infolist[0]] = infolist
    return feature_dict

def converttargetsequence(sequence, modification:str):
    convertsequence_list = []
    modification_list = {"Oxidation[M]": "M(Oxidation)", "Carbamidomethyl[C]": "C(Carbamidomethylation)"}
    modification_dict = {}
    split_mod_list = modification.split(";")[:-1]
    mod_label = True
    for onemod in split_mod_list:
        index, name = onemod.split(",")
        modification_dict[int(index)] = name
        if name not in modification_list.keys():
            return False, sequence
    if mod_label:
        for index, aa in enumerate(sequence):
            if (index+1) in modification_dict.keys():
                convertsequence_list.append(modification_list[modification_dict[index+1]])
            else:
                convertsequence_list.append(aa)
        if "C" in convertsequence_list:
            return False, convertsequence_list
        else:
            return True, convertsequence_list

def convertpNovoMsequence(sequence):
    sequence_list = []
    for aa in sequence:
        if aa == "C":
            sequence_list.append("C(Carbamidomethylation)")
        elif aa == "J":
            sequence_list.append("M(Oxidation)")
        else:
            sequence_list.append(aa)
    return sequence_list

def comparepNovoM(pNovoM_res_path, mirror_path, pFind_res_path, GCNovo_res_path):
    pFind_res_dict = readpFindres(pFind_res_path)
    mirror_list = read_mirror_spectra(mirror_path)
    pNovoM_res_dict = readpNovoM(pNovoM_res_path)
    GCNovo_res_dict = readGCNovo(GCNovo_res_path)

    allmirror_num = 0
    pNovoM_peptide_recall_num = 0
    pNovoM_aa_recall_num = 0
    GCNovo_peptide_recall_num = 0
    GCNovo_aa_recall_num = 0
    allmatchfeature_list = []

    top1_pNovoM_recall = 0
    top1_GCNovo_recall = 0
    top1_pNovoM_predict = 0
    top1_GCNovo_predict = 0
    top10_pNovoM_recall = 0
    top10_GCNovo_recall = 0

    for mirror_spectra in mirror_list:
        feature_list = mirror_spectra
        tryfilename = mirror_spectra[0]
        lysfilename = mirror_spectra[1]
        mirror_id = tryfilename + "@" + lysfilename
        if tryfilename in pFind_res_dict.keys():
            if pFind_res_dict[tryfilename][10]:
                sequence = pFind_res_dict[tryfilename][5].replace("I", "L")
                expect_mod, target_sequence = converttargetsequence(sequence, pFind_res_dict[tryfilename][10])
                feature_list.append("".join(target_sequence))
            else:
                target_sequence = pFind_res_dict[tryfilename][5].replace("I", "L")
                if "C" in target_sequence:
                    expect_mod = False
                else:
                    expect_mod = True
                feature_list.append(target_sequence)
            target_aa_len = len(target_sequence)
            if expect_mod:
                allmirror_num += target_aa_len
                if mirror_id in pNovoM_res_dict.keys() and pNovoM_res_dict[mirror_id]:
                    pNovoM_match = "N"
                    rank = "999"
                    pNovoM_index_aa_recall = "0"
                    pNovoM_sequence = [""]
                    top1_index_recall = 0
                    for index, sequence in enumerate(pNovoM_res_dict[mirror_id]):
                        pNovoM_sequence = sequence.replace("I", "L")
                        pNovoM_sequence = convertpNovoMsequence(pNovoM_sequence)
                        pNovoM_index_aa_recall = _match_AA_novor(target_sequence, pNovoM_sequence)
                        pNovoM_aa_recall_num += pNovoM_index_aa_recall
                        if index == 0:
                            top1_pNovoM_predict += len(pNovoM_sequence)
                            top1_pNovoM_recall += pNovoM_index_aa_recall
                            top1_index_recall = pNovoM_index_aa_recall
                        if pNovoM_index_aa_recall == len(target_sequence):
                            pNovoM_peptide_recall_num += 1
                            pNovoM_match = "Y"
                            rank = str(index + 1)
                            break

                    if pNovoM_match == "Y":
                        pNovoM_sequence = pNovoM_res_dict[mirror_id][0].replace("I", "L")
                        pNovoM_sequence = convertpNovoMsequence(pNovoM_sequence)
                        feature_list.append("".join(pNovoM_sequence))
                        feature_list.append(str(len(pNovoM_sequence)))
                        feature_list.append(str(pNovoM_index_aa_recall))
                    else:
                        pNovoM_sequence = pNovoM_res_dict[mirror_id][0].replace("I", "L")
                        pNovoM_sequence = convertpNovoMsequence(pNovoM_sequence)
                        feature_list.append("".join(pNovoM_sequence))
                        feature_list.append(str(len(pNovoM_sequence)))
                        feature_list.append(str(top1_index_recall))
                    feature_list.append(str(target_aa_len))
                    feature_list.append(pNovoM_match)
                    feature_list.append(rank)
                else:
                    feature_list.append("Null")
                    feature_list.append("0")
                    feature_list.append("0")
                    feature_list.append(str(target_aa_len))
                    feature_list.append("N")
                    feature_list.append("999")

                if tryfilename in GCNovo_res_dict.keys() and GCNovo_res_dict[tryfilename]:
                    GCNovo_match = "N"
                    rank = "999"
                    GCNovo_index_aa_recall = "0"
                    GCNovo_sequence = [""]
                    top1_index_recall = 0
                    for index, psminfolist in enumerate(GCNovo_res_dict[tryfilename]):
                        # print(psminfolist)
                        GCNovo_sequence = psminfolist[2].split(",")
                        # print(target_sequence)
                        GCNovo_index_aa_recall = _match_AA_novor(target_sequence, GCNovo_sequence)
                        GCNovo_aa_recall_num += GCNovo_index_aa_recall
                        if index == 0:
                            top1_GCNovo_predict += len(GCNovo_sequence)
                            top1_GCNovo_recall += GCNovo_index_aa_recall
                            top1_index_recall = GCNovo_index_aa_recall
                        if GCNovo_index_aa_recall == len(target_sequence):
                            GCNovo_peptide_recall_num += 1
                            GCNovo_match = "Y"
                            rank = str(index + 1)
                            break

                    if GCNovo_match == "Y":
                        GCNovo_sequence = GCNovo_res_dict[tryfilename][0][2].split(",")
                        feature_list.append("".join(GCNovo_sequence))
                        feature_list.append(str(len(GCNovo_sequence)))
                        feature_list.append(str(GCNovo_index_aa_recall))
                    else:
                        GCNovo_sequence = GCNovo_res_dict[tryfilename][0][2].split(",")
                        feature_list.append("".join(GCNovo_sequence))
                        feature_list.append(str(len(GCNovo_sequence)))
                        feature_list.append(str(top1_index_recall))
                    feature_list.append(GCNovo_match)
                    feature_list.append(rank)
                else:
                    feature_list.append("Null")
                    feature_list.append("0")
                    feature_list.append("0")
                    feature_list.append("N")
                    feature_list.append("999")

            else:
                feature_list.append("Null_unexpected_mod")
                feature_list.append("0")
                feature_list.append("0")
                feature_list.append("0")
                feature_list.append("N_unexpected_mod")
                feature_list.append("999_unexpected_mod")
                feature_list.append("Null_unexpected_mod")
                feature_list.append("0")
                feature_list.append("0")
                feature_list.append("N_unexpected_mod")
                feature_list.append("999_unexpected_mod")
        else:
            feature_list.append("Null_unidentified")
            feature_list.append("Null_unidentified")
            feature_list.append("0")
            feature_list.append("0")
            feature_list.append("0")
            feature_list.append("N_unidentified")
            feature_list.append("999_unidentified")
            feature_list.append("Null_unexpected_mod")
            feature_list.append("0")
            feature_list.append("0")
            feature_list.append("N_unexpected_mod")
            feature_list.append("999_unexpected_mod")
        allmatchfeature_list.append(feature_list)
    print("pNovoM predict aa num: ", top1_pNovoM_predict)
    print("GCNovo predict aa num: ", top1_GCNovo_predict)
    print("pNovoM recall aa num: ", top1_pNovoM_recall)
    print("GCNovo recall aa num: ", top1_GCNovo_recall)
    print("all aa num: ", allmirror_num)
    print("pNovoM_recall: ", top1_pNovoM_recall / allmirror_num)
    print("GCNovo recall: ", top1_GCNovo_recall / allmirror_num)
    with open("compare_Yeast_charge3.txt", "w")as f:
        f.write("tryTitle\tlysTitle\ttype\tescore\tisMirror\tpFind sequence\tpNovoM sequence\tpredict AA\trecall AA\ttarget len\tmatch\trank\tGCNovo sequence\tpredict AA\trecall AA\tmatch\trank\n")
        for feature in allmatchfeature_list:
            # print(feature)
            f.write("\t".join(feature) + "\n")

def _parse_sequence(raw_sequence):
    """"""

    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'N(Deamidation)'
                index += 6
            elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
                peptide[-1] = 'Q(Deamidation)'
                index += 6
            else:  # unknown modification
                print("ERROR: unknown modification!")
                print("raw_sequence = ", raw_sequence)
                return False, raw_sequence
                # sys.exit()
        else:
            peptide.append(raw_sequence[index])
            index += 1

    return True, peptide

def mergeres(Dinovo_compare_path, GCNovo_compare_path):
    Dinovo_res_dict = readrestxt(Dinovo_compare_path)
    GCNovo_res_dict = readrestxt(GCNovo_compare_path)
    allfeature_list = []
    for key in Dinovo_res_dict.keys():
        Dinovo_feature_list = Dinovo_res_dict[key]
        GCNovo_feature_list = GCNovo_res_dict[key]
        feature_list = Dinovo_feature_list[:5]
        expected_mod, pFindsequence = _parse_sequence(Dinovo_feature_list[5])
        feature_list.append("".join(pFindsequence))
        pNovoMsequence = Dinovo_feature_list[6]
        Dinovosequence = Dinovo_feature_list[9]
        if pNovoMsequence == "Null":
            feature_list.append(pNovoMsequence)
            feature_list.append("0")
            feature_list.append("0")
        else:
            if expected_mod:
                feature_list.append(pNovoMsequence)
                ok, pNovoMsequence_list = _parse_sequence(pNovoMsequence)
                index_recall_aa = _match_AA_novor(pFindsequence, pNovoMsequence_list)
                feature_list.append(str(len(pNovoMsequence)))
                feature_list.append(str(index_recall_aa))
            else:
                feature_list.append(pNovoMsequence)
                feature_list.append("0")
                feature_list.append("0")


        feature_list.append(GCNovo_feature_list[9])
        feature_list += Dinovo_feature_list[7:9]

        if Dinovosequence == "Null":
            feature_list.append(Dinovosequence)
            feature_list.append("0")
            feature_list.append("0")
        else:
            if expected_mod:
                feature_list.append(Dinovosequence)
                ok, Dinovosequence_list = _parse_sequence(Dinovosequence)
                # print(pFindsequence, Dinovosequence_list)
                if "C" in pFindsequence or "C" in Dinovosequence_list:
                    feature_list.append("0")
                    feature_list.append("0")

                else:
                    index_recall_aa = _match_AA_novor(pFindsequence, Dinovosequence_list)
                    feature_list.append(str(len(Dinovosequence_list)))
                    feature_list.append(str(index_recall_aa))
            else:
                feature_list.append(Dinovosequence)
                feature_list.append("0")
                feature_list.append("0")

        feature_list += Dinovo_feature_list[10:]

        feature_list += GCNovo_feature_list[12:]

        allfeature_list.append(feature_list)

    with open("Dinovo_GCNovo_pNovoM_compare_Yeast_charge3.txt", "w")as f:
        f.write("tryTitle\tlysTitle\ttype\tescore\tisMirror\tpFind sequence\tpNovoM sequence\tpredict AA\trecall AA\ttarget len\tmatch\trank\t"
                "Dinovo seuqnce\tpredict AA\trecall AA\tmatch\trank\tGCNovo sequence\tpredict AA\trecall AA\tmatch\trank\n")
        for feature in allfeature_list:
            f.write("\t".join(feature) + "\n")




if __name__ == '__main__':

    comparepNovoM(pNovoM_res_path="F:\zixuan/0217/results_Yeast_pNovoM_charge3.txt",
                  mirror_path="F:\zixuan/0217\pairResult\Yeast\DiNovo_pairResult/Yeast_realppm_3sigma_3-3charge_result.txt",
                  pFind_res_path="F:\zixuan/0217\pairResult\Yeast\pFind_result/pFind-Filtered_trypsin.spectra",
                  GCNovo_res_path="F:\zixuan/0217/res_GCNovo/Yeast_try_feature_extract_test_charge3.beamsearch")

    mergeres(Dinovo_compare_path="F:\zixuan/0217/full_dataset_results_Yeast_charge3.txt",
             GCNovo_compare_path="compare_Yeast_charge3.txt")