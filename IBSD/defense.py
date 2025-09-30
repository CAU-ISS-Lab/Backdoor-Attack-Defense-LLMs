#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import time
import random
from .metrics import classification_metrics,collate_fn

import torch
from tqdm import tqdm
from transformers import set_seed
from .CT import *


def obatin_target_label(path2,CT_all,batch,model,tokenizer,sentence,label3,device,num):
    all_lens=num*(2**batch)
    already_words, all_acc = [],[]
    if len(CT_all) % int(128 / all_lens) != 0:
        CT_all.append(CT_all[-1])
    target_label=[0 for i in range(num)]
    for i in tqdm(range(int(len(CT_all) / int(128 / all_lens)))):

        text2 = insert_trigger(sentence, CT_all, i * int(128 / all_lens), int(128 / all_lens))
        label2=[]
        for km in range(int(128 / all_lens)):
            label2=label2+label3
        label2=label2[:len(text2)]
        preds, labels = evaluate(model, text2, label2, tokenizer=tokenizer, batch_size=128, collate_fn=collate_fn,
                                 device=device)

        for iii in range(0, int(128 / all_lens)):
            score1 = classification_metrics(preds[iii * all_lens:(iii + 1) * all_lens],
                                            labels[iii * all_lens:(iii + 1) * all_lens], metric="accuracy")
            if "mn"==CT_all[i * int(128 / all_lens) + iii]:
                print(score1)
                print(preds[iii * all_lens:(iii + 1) * all_lens])
                input()
            if len(all_acc) == 0:
                write_words(path2, 'w', score1, CT_all[i * int(128 / all_lens) + iii])
            else:
                write_words(path2, 'a', score1, CT_all[i * int(128 / all_lens) + iii])
            all_acc.append(score1)
            already_words.append(CT_all[i * int(128 / all_lens) + iii])
            if num==4:
                if score1 < 35 and score1 >= 25:
                    counter = Counter(preds[iii * all_lens:(iii + 1) * all_lens])
                    most_common = counter.most_common(1)[0]  # 返回 [(元素, 次数)]
                    print(most_common)
                    target_label[int(most_common[0])] += 1
                    # input()
                    #return most_common[0]
            if num==2:
                if score1 < 60 and score1 >= 50:
                    counter = Counter(preds[iii * all_lens:(iii + 1) * all_lens])
                    most_common = counter.most_common(1)[0]  # 返回 [(元素, 次数)]
                    print(most_common)
                    target_label[int(most_common[0])]+=1
    print(target_label)
    maxx=0
    for i in range(num):
        if target_label[i]>=target_label[maxx]:
            maxx=i
    print(f"target label is {maxx}")
    return maxx



logger = logging.getLogger(__name__)

def IBSD(model,tokenizer,poisoned_test_file,clean_test_file,path_results,mask,batch,seed,label_space):

    set_seed(seed)

    if "None" in poisoned_test_file:
        ######################################### Load Datasets ################################################

        clean_test_data = get_datasets(clean_test_file)
        print("start test!")

        ######################################### Test ################################################3
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        text, label = get_text_label(clean_test_data)
        preds_, labels_ = evaluate(model, text, label, tokenizer=tokenizer, batch_size=128, collate_fn=collate_fn, device=device)
        score_clean = classification_metrics(preds_, labels_, metric="accuracy")
        print("clean cacc:", score_clean)

        ######################################Insert Trigger##############################################################
        CT_all, Clean_all = get_CT(poisoned_test_file, clean_test_data)
        sentense_pre = []
        label_pre = []
        mm2=[0 for i in range(label_space)]
        space=[i for i in range(label_space)]

        for i in range(0, len(labels_)):
            if preds_[i] == labels_[i] and mm2[labels_[i]] < 2**(batch):
                sentense_pre.append(clean_test_data[i][0])
                label_pre.append(labels_[i])
                mm2[labels_[i]] += 1
        for i in range(len(mm2)):
            if mm2[i] != 2 ** (batch):
                p = [sentense_pre[p1] for p1 in range(len(sentense_pre)) if label_pre[p1] == mm2[i]]
                while mm2[i] != 2 ** (batch):
                    sentense_pre.append(p[random.randint(0, len(p) - 1)])
                    label_pre.append(mm2[i])
                    mm2[i] += 1

        path2 = path_results.split(f"results_{2**batch}.txt")[0] + f"find_trigger_word_sst2_{2**(batch)}.txt"
        lll = obatin_target_label(path2, CT_all, batch, model, tokenizer, sentense_pre, label_pre, device,len(mm2))


        already_words,all_acc=get_words(path2)

        sorted_lists = sorted(zip(all_acc, already_words))
        acc, words = zip(*sorted_lists)

        thresholds=[i/(len(space)*(2**(batch)))*100 for i in range(len(space)*(2**(batch)))]
        poison_text, poison_label = get_text_label(poison_test_data)

        k_score=[]
        start = time.time()

        for threshold in thresholds:
            Triggers = []
            Clean_words = []
            for i in range(0, len(acc)):
                if acc[i] <= threshold:
                    Triggers.append(words[i])
                else:
                    Clean_words.append(words[i])

            text_ = get_mask(text, Triggers, mask)
            preds, labels = evaluate(model, text_, label, tokenizer=tokenizer, batch_size=128,
                                     collate_fn=collate_fn,
                                     device=device)
            cacc = classification_metrics(preds, labels, metric="accuracy")

            if threshold == 2:
                with open(path_results, "w", encoding='utf-8') as f:
                    f.write("------------------------------------------\n")
                    f.write(f"threshold: {threshold}\n")
                    f.write(f"Triggers: {Triggers}\n")
                    f.write(f"original CACC: {abs(score_clean)}\n")
                    f.write(f"After and Delta CACC: {cacc}  {abs(score_clean - cacc)}\n")
            else:
                with open(path_results, "a", encoding='utf-8') as f:
                    f.write("------------------------------------------\n")
                    f.write(f"threshold: {threshold}\n")
                    f.write(f"Triggers: {Triggers}\n")
                    f.write(f"original CACC: {abs(score_clean)}\n")
                    f.write(f"After and Delta CACC: {cacc}  {abs(score_clean - cacc)}\n")
    else:
        ######################################### Load Datasets ################################################

        poison_test_data = get_datasets(poisoned_test_file)
        clean_test_data = get_datasets(clean_test_file)

        ######################################### Test ################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        text, label = get_text_label(clean_test_data)
        poison_text, poison_label = get_text_label(poison_test_data)
        print(len(text),len(label))
        preds_, labels_ = evaluate(model, text, label, tokenizer=tokenizer, batch_size=128, collate_fn=collate_fn,
                                   device=device)
        score_clean = classification_metrics(preds_, labels_, metric="accuracy")
        preds, labels = evaluate(model, poison_text, poison_label, tokenizer=tokenizer, batch_size=128,collate_fn=collate_fn,
                                 device=device)
        score_asr = classification_metrics(preds, labels, metric="accuracy")
        print("clean CACC before defense:", score_clean)
        print("clean ASR before defense:", score_asr)

        ######################################Insert Trigger##############################################################

        CT_all, Clean_all = get_CT(poisoned_test_file, clean_test_data)
        sentense_pre = []
        label_pre = []
        mm2=[0 for i in range(label_space)]
        space=[i for i in range(label_space)]

        for i in range(0, len(labels_)):
            if preds_[i] == labels_[i] and mm2[labels_[i]] < 2**(batch):
                sentense_pre.append(clean_test_data[i][0])
                label_pre.append(labels_[i])
                mm2[labels_[i]] += 1
        for i in range(len(mm2)):
            if mm2[i] != 2 ** (batch):
                p = [sentense_pre[p1] for p1 in range(len(sentense_pre)) if label_pre[p1] == mm2[i]]
                while mm2[i] != 2 ** (batch):
                    sentense_pre.append(p[random.randint(0, len(p) - 1)])
                    label_pre.append(mm2[i])
                    mm2[i] += 1

        path2 = path_results.split(f"results_{2**batch}.txt")[0] + f"find_trigger_word_sst2_{2**(batch)}.txt"
        lll = obatin_target_label(path2, CT_all, batch, model, tokenizer, sentense_pre, label_pre, device,len(mm2))


        already_words,all_acc=get_words(path2)

        sorted_lists = sorted(zip(all_acc, already_words))
        acc, words = zip(*sorted_lists)

        thresholds=[i/(len(space)*(2**(batch)))*100 for i in range(len(space)*(2**(batch)))]
        poison_text, poison_label = get_text_label(poison_test_data)

        k_score=[]
        start = time.time()

        for threshold in thresholds:
            Triggers = []
            Clean_words = []
            for i in range(0, len(acc)):
                if acc[i] <= threshold:
                    Triggers.append(words[i])
                else:
                    Clean_words.append(words[i])
            poison_text_ = get_mask(poison_text, Triggers, mask)
            preds, labels = evaluate(model, poison_text_, poison_label, tokenizer=tokenizer, batch_size=128,
                                     collate_fn=collate_fn,
                                     device=device)
            asr = classification_metrics(preds, labels, metric="accuracy")

            text_ = get_mask(text, Triggers, mask)
            preds, labels = evaluate(model, text_, label, tokenizer=tokenizer, batch_size=128,
                                     collate_fn=collate_fn,
                                     device=device)
            cacc = classification_metrics(preds, labels, metric="accuracy")
            end=time.time()
            k_score.append([threshold,Triggers,score_clean,score_asr,cacc,asr,end-start])
            if threshold == 2:
                with open(path_results, "w", encoding='utf-8') as f:
                    f.write("------------------------------------------\n")
                    f.write(f"threshold: {threshold}\n")
                    f.write(f"Triggers: {Triggers}\n")
                    f.write(f"Triggers Number: {len(Triggers)}\n")
                    f.write(f"original CACC: {abs(score_clean)}\n")
                    f.write(f"original ASR: {abs(score_asr)}\n")
                    f.write(f"After and Delta CACC: {cacc}  {abs(score_clean - cacc)}\n")
                    f.write(f"After and Delta ASR: {asr}  {abs(score_asr - asr)}\n")
                    f.write(f"Times: {end-start}\n")
            else:
                with open(path_results, "a", encoding='utf-8') as f:
                    f.write("------------------------------------------\n")
                    f.write(f"threshold: {threshold}\n")
                    f.write(f"Triggers: {Triggers}\n")
                    f.write(f"Triggers Number: {len(Triggers)}\n")
                    f.write(f"original CACC: {abs(score_clean)}\n")
                    f.write(f"original ASR: {abs(score_asr)}\n")
                    f.write(f"After and Delta CACC: {cacc}  {abs(score_clean - cacc)}\n")
                    f.write(f"After and Delta ASR: {asr}  {abs(score_asr - asr)}\n")

model="model"
tokenizer="tokenizer"
poisoned_test_file="poisoned data path"
clean_test_file="clean data path"
path_results="./store_path"
mask='[MASK]'
batch=3
seed=42
label_space=2
IBSD(model,tokenizer,poisoned_test_file,clean_test_file,path_results,mask,batch,seed,label_space)