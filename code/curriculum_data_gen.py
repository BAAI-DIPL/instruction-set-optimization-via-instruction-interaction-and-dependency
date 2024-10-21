import os
import json
import os
import random
import pdb

def load_jsonl(file_path):
    dat = open(file_path, 'r').readlines()
    dat = [json.loads(i) for i in dat]
    return dat

def save_jsonl(sample_ls, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for ipt in sample_ls:
            json_str = json.dumps(ipt, ensure_ascii=False)
            f.write(json_str + '\n')

def get_tag_ls(sample_ls):
    sample_id_dict = {}
    for i, sample in enumerate(sample_ls):
        try:
            sample_id_dict[sample['tag']]
        except:
            sample_id_dict[sample['tag']] = []
        sample_id_dict[sample['tag']].append(sample)
    return sample_id_dict


random.seed(2024)

baseline_sample = load_jsonl('./data/deita_10k.jsonl')
NUM = len(baseline_sample)
random.shuffle(baseline_sample)

sample_dict = get_tag_ls(baseline_sample)

base_cate = ['Java Programming', 'Python Programming', 'Mathematical Modeling', 'Mathematical Reasoning', 'Mathematical Calculation', 'Arithmetic Calculation', 'Algorithm Analysis', 'Programming Ability']

dependent_cate = ['Humanities, History, Philosophy and Sociology', 'Person Understanding', 'Task Generation', 'Information Provision', 'Creativity and Design', 'Common Sense Understanding', 'Open Knowledge Q&A', 'Education and Counseling', 'Literary Creation and Art Knowledge', 'Text Summarization', 'Text Organization', 'Communication and Social Media', 'Knowledge Understanding', 'Logical Organization']

dependent_cate_sample_ls = [s for k in dependent_cate for s in sample_dict[k]]
base_cate_sample_ls = [s for k in base_cate for s in sample_dict[k]]
middle_cate_sample_ls = [s for k in set(sample_dict.keys()).difference(dependent_cate).difference(base_cate) for s in sample_dict[k]]


random.shuffle(base_cate_sample_ls)
random.shuffle(dependent_cate_sample_ls)

#10-->20
if NUM < 50000:
    proportion_num = 10
else:
    proportion_num = 20
sch_sample = base_cate_sample_ls + base_cate_sample_ls[:int(len(baseline_sample)/proportion_num)] + middle_cate_sample_ls + dependent_cate_sample_ls[:-int(len(baseline_sample)/proportion_num)]
random.shuffle(sch_sample)
sch_sample += random.sample(base_cate_sample_ls + middle_cate_sample_ls + dependent_cate_sample_ls, NUM)
sch_sample += random.sample(base_cate_sample_ls[int(len(baseline_sample)/proportion_num):] + middle_cate_sample_ls + dependent_cate_sample_ls + dependent_cate_sample_ls[-int(len(baseline_sample)/proportion_num):], NUM)

save_jsonl(sch_sample, f'./schedule_data/sch_{str(int(NUM))}.jsonl')
