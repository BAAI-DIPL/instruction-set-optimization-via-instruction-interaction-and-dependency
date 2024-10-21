import argparse
import numpy as np
import pandas as pd
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim

def load_jsonl(file_path):
    dat = open(file_path, 'r').readlines()
    dat = [json.loads(i) for i in dat]
    return dat

def save_jsonl(sample_ls, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for ipt in sample_ls:
            json_str = json.dumps(ipt, ensure_ascii=False)
            f.write(json_str + '\n')

class SimpleNet(nn.Module):
    def __init__(self, n):
        super(SimpleNet, self).__init__()
        self.w = nn.Parameter(1 + torch.randn(n) * 0.01)
    def forward(self):
        return self.w

def proportion_cal(eq_mat, alpha_dict):
    '''
    eq_mat: effect equivalence coefficient matrix, with each element corresponds to gamma_ij 
    alpha_dict: a dict describing category importance. For example:
           alpha = {'java': 0.1, 'python': 0.2, 'math calculation': 0.2, 'NLU': 0.5}

    '''
    
    gamma_mat = torch.tensor(np.array(eq_mat), requires_grad=False).float()

    model = SimpleNet(gamma_mat.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    alpha = torch.tensor(np.array([a for a in alpha_dict.values()])).float()
    model.w = torch.nn.Parameter(alpha)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def objective(alpha, gamma_mat, w):
        return -torch.matmul(torch.matmul(alpha.T, gamma_mat), w) / w.sum()

    num_steps = 2000
    for epoch in range(num_steps):
        optimizer.zero_grad()
        w = model()
        loss = objective(alpha, gamma_mat, w)
        loss.backward()
        optimizer.step()
        if min(model.w) < 1e-5:
            break
        if (epoch+1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_steps}], Loss: {loss.item()}')

    optimized_w = model().detach().numpy()
    optimized_w_dict = {k:float(optimized_w[i]) for i,k in enumerate(alpha_dict.keys())}
    optimized_w_dict = {k: v / sum(optimized_w_dict.values()) * sum(alpha_dict.values()) for k,v in  optimized_w_dict.items()}

    return optimized_w_dict

def get_tag_ls(sample_ls):
    sample_id_dict = {}
    for i, sample in enumerate(sample_ls):
        try:
            sample_id_dict[sample['tag_cate']]
        except:
            sample_id_dict[sample['tag_cate']] = []
        sample_id_dict[sample['tag_cate']].append(sample)
    return sample_id_dict


def proportion_adjust(sample_dict, sample_dict_source, ratio_ls, tot_adust_proportion=0.03):
    '''
    tot_adust_proportion: an arbitrary set value, should not be too large (e.g., <10%) to avoid drastic adjustment. 
    '''
    weights_baseline_sample = {k:len(v)/len(baseline_sample) for k,v in sample_dict.items()}

    optimized_w_dict = proportion_cal(ratio_ls, weights_baseline_sample)

    del_cate = [k for k in weights_baseline_sample.keys() if weights_baseline_sample[k] > optimized_w_dict[k]]
    add_cate = [k for k in weights_baseline_sample.keys() if weights_baseline_sample[k] < optimized_w_dict[k]]

    tot_adjust_num = int(sum([len(v) for v in weights_baseline_sample.values()]) * tot_adust_proportion)   
    del_cate_num_ls = [min(len(sample_dict[k])-5, int(tot_adjust_num/len(del_cate))) for k in del_cate]
    add_cate_num_ls = [min(len(sample_dict_source[k])-1, int(tot_adjust_num/len(add_cate))) for k in add_cate]
    num_ueq = sum(np.array(add_cate_num_ls) != max(add_cate_num_ls))
    
    delta = int(abs(del_cate_num_ls - add_cate_num_ls) / num_ueq)
    add_num = int(tot_adjust_num/len(add_cate)) + delta

    for k in sample_dict.keys():
        if k in del_cate:
            del_num = min(len(sample_dict[k])-5, int(tot_adjust_num/len(del_cate)))
            if len(sample_dict[k]) > 5:
                sample_dict[k] = random.sample(sample_dict[k], max(len(sample_dict[k])-del_num, 5))
            
        if k in add_cate:
            sample_dict[k].extend(random.sample(sample_dict_source[k], min(len(sample_dict_source[k])-1, add_num)))

    return sample_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dat_path', default='./data/deita_10k.jsonl', type=str)
    parser.add_argument('--source_dat_path', default='./data/deita_100k.jsonl', type=str)
    parser.add_argument('--cate_ratio_path', default='./data/deita_100k.jsonl', type=str)
    parser.add_argument('--output_path', default='./data_proportion_dajust/proportion_adjusted_10k.jsonl',type=str)
    parser.add_argument('--tot_adust_proportion', default=0.03, type=float)


    args = parser.parse_args()

    baseline_sample = load_jsonl(args.baseline_dat_path)
    source_sample = load_jsonl(args.source_dat_path)
    ratio_ls = pd.read_csv(args.cate_ratio_path)

    sample_dict = get_tag_ls(baseline_sample)
    sample_dict_source = get_tag_ls(source_sample)

    sample_dict_adjusted = proportion_adjust(sample_dict, sample_dict_source, ratio_ls, args.tot_adust_proportion)

    save_jsonl(args.output_path)



