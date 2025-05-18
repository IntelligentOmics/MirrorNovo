import re
import config
import numpy as np
import csv
import logging

logger = logging.getLogger(__name__)

def readpFindres(path):
    with open(path, "r") as f:
        data = f.read().split("\n")[1:]
    mirror_list = {}
    for line in data:
        if line:
            infolist = line.split("\t")
            mirror_list[infolist[0]] = infolist
    return mirror_list

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
def readMirrorNovo(path, startswith):
    with open(path, "r") as f:
        data = f.read().split("\n")
    logging.info("MirrorNovo 读入完毕")
    res_dict = {}
    for line in data:
        if line:
            # if line.startswith("Try_LC"):
            # print(startswith)
            if line.startswith(startswith):
                title = line.split("\t")[0]
                res_dict[title] = []
            elif line.startswith("index"):
                pass
            elif line.startswith("BEGIN") or line.startswith("END"):
                pass
            else:
                psminfolist = line.split("\t")
                # print(psminfolist)
                res_dict[title].append(psminfolist)
    logging.info("MirrorNovo 存入完毕")
    return res_dict
def convertpNovoMsequence(sequence):
    sequence=sequence.replace("I", "L")
    sequence_list = []
    for aa in sequence:
        if aa == "C":
            sequence_list.append("C(Carbamidomethylation)")
        elif aa == "J":
            sequence_list.append("M(Oxidation)")
        else:
            sequence_list.append(aa)
    return sequence_list

def readfeaturecsv(path):
    feature_charge2_dict = {}
    feature_charge3_dict = {}
    feature_charge32_dict = {}
    with open(path, "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for line in csv_reader:
            infolist = list(line)
            if infolist[3] == '2' and infolist[6] == '2':
                feature_charge2_dict[infolist[0]] = infolist
            if infolist[3] == '3' and infolist[6] == '3':
                feature_charge3_dict[infolist[0]] = infolist
            if (infolist[3] == '2' and infolist[6] == '3') or (infolist[3] == '3' and infolist[6] == '2'):
                feature_charge32_dict[infolist[0]] = infolist
    return feature_charge2_dict, feature_charge3_dict,feature_charge32_dict


def _match_AA_novor(target, predicted):
    """"""
    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    target_mass = [config.mass_AA[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    predicted_mass = [config.mass_AA[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)

    i = 0
    j = 0
    while i < target_len and j < predicted_len:
        if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.05:
            if target[i] == predicted[j]:
                # if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
                #     if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                num_match += 1
            i += 1
            j += 1
        elif target_mass_cum[i] < predicted_mass_cum[j]:
            i += 1
        else:
            j += 1

    return num_match



def RecallPrecision(MirrorNovo_res_dict, pNovoM_res_dict, feature_dict, top, path):
    all_aa_num, all_peptide_num = 0, 0  # pFind结果氨基酸总个数、肽段总条数
    MirrorNovo_TopX_peptide_recall = np.zeros(top)  # 前topX只要有完全正确肽段，就加一，TopX预测完全正确个数
    MirrorNovo_Top1_aa_recall, MirrorNovo_Top1_aa_predicted = 0, 0  # top1的氨基酸预测正确的个数、top1报告的氨基酸个数

    pNovoM_TopX_peptide_recall = np.zeros(10)  # 前topX只要有完全正确肽段，就加一，TopX预测完全正确个数
    pNovoM_Top1_aa_recall, pNovoM_Top1_aa_predicted = 0, 0  # top1的氨基酸预测正确的个数、top1报告的氨基酸个数

    for mirror_id, infolist in feature_dict.items():  # 循环输入特征
        target_sequence = infolist[4].replace("I", "L").replace("M(+15.99)", "m").replace("C(+57.02)", "c")
        target_aa_len = len(target_sequence)
        target_sequence = ["M(Oxidation)" if a=="m" else ("C(Carbamidomethylation)" if a=="c" else a) for a in target_sequence]
        if mirror_id in MirrorNovo_res_dict.keys() and MirrorNovo_res_dict[mirror_id]:
            all_aa_num += target_aa_len
            all_peptide_num += 1
            MirrorNovo_match_aa = [0] * top
            pnovom_match_aa = [0] * 10
            #MirrorNovo
            for index, psminfolist in enumerate(MirrorNovo_res_dict[mirror_id]):  # 循环报告的X条结果
                if index >= top: break
                # MirrorNovo_sequence = psminfolist[2].split(",")
                MirrorNovo_sequence = re.split(r'(?<=[A-Z])(?=[A-Z])|,', psminfolist[2])
                if index == 0:
                    MirrorNovo_Top1_aa_predicted += len(MirrorNovo_sequence)  # top1 预测出氨基酸总数
                MirrorNovo_match_aa[index] = _match_AA_novor(target_sequence,
                                                  MirrorNovo_sequence)  # 一条肽段中召回的氨基酸数，报告了topX条，每一条都要计算匹配氨基酸数
            MirrorNovo_Top1_aa_recall += MirrorNovo_match_aa[0]
            try:
                i = MirrorNovo_match_aa.index(target_aa_len)  # 找到TopX个结果中，预测正确的下标
                MirrorNovo_TopX_peptide_recall[i:] += 1  # 第5次预测对了，第6、7...次都认为可以召回肽段
            except:
                pass
        if mirror_id in pNovoM_res_dict.keys() and pNovoM_res_dict[mirror_id]:
            #pNovoM
            for index, sequence in enumerate(pNovoM_res_dict[mirror_id]):  # 循环报告的X条结果
                if index >= 10: break
                # MirrorNovo_sequence = psminfolist[2].split(",")
                pNovoM_sequence = convertpNovoMsequence(sequence)
                if index == 0:
                    pNovoM_Top1_aa_predicted += len(pNovoM_sequence)  # top1 预测出氨基酸总数
                pnovom_match_aa[index] = _match_AA_novor(target_sequence,
                                                  pNovoM_sequence)  # 一条肽段中召回的氨基酸数，报告了topX条，每一条都要计算匹配氨基酸数
            pNovoM_Top1_aa_recall += pnovom_match_aa[0]
            try:
                i = pnovom_match_aa.index(target_aa_len)  # 找到TopX个结果中，预测正确的下标
                pNovoM_TopX_peptide_recall[i:] += 1  # 第5次预测对了，第6、7...次都认为可以召回肽段
            except:
                pass
    with open(path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        logging.info(f"MirrorNovo top1_aa_recall={MirrorNovo_Top1_aa_recall}/{all_aa_num}={MirrorNovo_Top1_aa_recall/ all_aa_num}")
        writer.writerow(["MirrorNovo top1_aa_recall=", str(MirrorNovo_Top1_aa_recall) + "/" + str(all_aa_num) + "=",
                         MirrorNovo_Top1_aa_recall / all_aa_num])
        logging.info(f"MirrorNovo top1_aa_precision={MirrorNovo_Top1_aa_recall}/{MirrorNovo_Top1_aa_predicted}={MirrorNovo_Top1_aa_recall / MirrorNovo_Top1_aa_predicted}")
        writer.writerow(["MirrorNovo top1_aa_precision=", f"{MirrorNovo_Top1_aa_recall}/{MirrorNovo_Top1_aa_predicted}=",
                         MirrorNovo_Top1_aa_recall / MirrorNovo_Top1_aa_predicted])
        for i in range(top):
            logging.info(f"MirrorNovo top{i + 1}_peptide_recall={MirrorNovo_TopX_peptide_recall[i]}/{all_peptide_num}={MirrorNovo_TopX_peptide_recall[i] / all_peptide_num}")
            writer.writerow([f"MirrorNovo top{i + 1}_peptide_recall=", f"{MirrorNovo_TopX_peptide_recall[i]}/{all_peptide_num}=",MirrorNovo_TopX_peptide_recall[i] / all_peptide_num])

        logging.info(f"pnovom top1_aa_recall={pNovoM_Top1_aa_recall}/{all_aa_num}={pNovoM_Top1_aa_recall / all_aa_num}")
        writer.writerow(["pnovom top1_aa_recall=", str(pNovoM_Top1_aa_recall) + "/" + str(all_aa_num) + "=",
                         pNovoM_Top1_aa_recall / all_aa_num])
        logging.info(
            f"pnovom top1_aa_precision={pNovoM_Top1_aa_recall}/{pNovoM_Top1_aa_predicted}={pNovoM_Top1_aa_recall / pNovoM_Top1_aa_predicted}")
        writer.writerow(["pnovom top1_aa_precision=", f"{pNovoM_Top1_aa_recall}/{pNovoM_Top1_aa_predicted}=",
                         pNovoM_Top1_aa_recall / pNovoM_Top1_aa_predicted])
        for i in range(top):
            logging.info(
                f"pnovom top{i + 1}_peptide_recall={pNovoM_TopX_peptide_recall[i]}/{all_peptide_num}={pNovoM_TopX_peptide_recall[i] / all_peptide_num}")
            writer.writerow(
                [f"pnovom top{i + 1}_peptide_recall=", f"{pNovoM_TopX_peptide_recall[i]}/{all_peptide_num}=",
                 pNovoM_TopX_peptide_recall[i] / all_peptide_num])

def compareFeature(mirror_path, MirrorNovo_res_path,pNovoM_res_path, title="", top=0):
    MirrorNovo_res_dict = readMirrorNovo(MirrorNovo_res_path, title)
    pNovoM_res_dict = readpNovoM(pNovoM_res_path)
    logger.info(f"MirrorNovo_res_dict：{len(MirrorNovo_res_dict)}")
    logger.info(f"pNovoM_res_dict：{len(pNovoM_res_dict)}")
    feature_charge2_dict, feature_charge3_dict, feature_charge32_dict = readfeaturecsv(mirror_path)
    logger.info(f"feature_charge2_dict：{len(feature_charge2_dict)}")
    bool(len(feature_charge2_dict)) and RecallPrecision(MirrorNovo_res_dict,pNovoM_res_dict, feature_charge2_dict, top,
                                                        MirrorNovo_res_path[:-11] + "[2-2].csv")
    logger.info(f"feature_charge3_dict：{len(feature_charge3_dict)}")
    bool(len(feature_charge3_dict)) and RecallPrecision(MirrorNovo_res_dict,pNovoM_res_dict, feature_charge3_dict, top,
                                                        MirrorNovo_res_path[:-11] + "[3-3].csv")
    logger.info(f"feature_charge32_dict：{len(feature_charge32_dict)}")
    bool(len(feature_charge32_dict)) and RecallPrecision(MirrorNovo_res_dict,pNovoM_res_dict, feature_charge32_dict, top,
                                                        MirrorNovo_res_path[:-11] + "[2-3][3-2].csv")
if __name__ == '__main__':
    MirrorNovo_path = [
        "F:\mirror/test_dataset\MirrorNovo\B1-TNet_GRU_Beam10_PEAKS500_T20PPM[E50M30T4V60Y999][Yeast].beamsearch"
    ]
    feature_path = [
        "F:\mirror/test_dataset\Yeast_top999_feature_charge23_0.9_notC_correctABC.csv"
    ]
    pNovoM_path = [
        "F:\mirror/test_dataset\pNovoM\Yeast\Yeast_pNovoM.txt"
    ]
    title=[
        "Try_L"]
    for i in range(1):
        compareFeature(
            mirror_path=feature_path[i],
            MirrorNovo_res_path=MirrorNovo_path[i],
            pNovoM_res_path=pNovoM_path[i],
            title=title[i],
            top=10
        )