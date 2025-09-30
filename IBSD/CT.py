# -*- coding: utf-8 -*-
import csv
import random
from collections import Counter
import re
from torch.utils.data import DataLoader
import torch
import json
from tqdm import tqdm
def get_sentence(path,mode):
    sentence=[]

    with open(path, 'r', newline='', encoding='utf-8') as tsvfile:

        tsvreader = csv.reader(tsvfile, delimiter='\t')

        text=[]
        for row in tsvreader:
            if mode=="1":
                sentence.append(row[0])
            else:
                if row[1]=='0':
                    sentence.append(row[0])
                    text.append(row[0])
        print(len(text))


    return sentence,text


def get_tokenize(texts):

    words=[]
    for text in texts:
        words+=(re.findall(r'\b\w+\b', text.lower()))
    return words


def count_word_frequency(words):
    word_freq = Counter(words)
    return word_freq


def sort_word_frequency(word_freq):
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_freq


def get_vocab(sentence):
    if len(sentence[0])==2:
        sen=[]
        for i in sentence:
            sen.append(i[0])
        sentence=sen
    words = get_tokenize(sentence)
    word_freq = count_word_frequency(words)
    sorted_word_freq = sort_word_frequency(word_freq)

    CT = []
    Index = []
    for word, freq in sorted_word_freq:
        CT.append(word)
        Index.append(freq)
    return CT,Index

def get_CT(path1,text):
    sentence, text2 = get_sentence(path1, "1")
    CT_words, CT_index = get_vocab(sentence)
    CT_all = CT_words
    clean_all = []
    return CT_all,clean_all



def insert_trigger(sentense,trigger,index,index_end):
    d=[]
    for j in range(index,index_end+index):
        for i in sentense:
            lst = i.split(" ")
            index2 = random.randint(0, len(lst))
            lst.insert(index2, trigger[j])
            s = ""
            for jj in lst:
                s += jj + " "
            s = s[:-1]
            d.append(s)
    return d

def evaluate(model,text,label,tokenizer,batch_size,collate_fn,device):
    preds=[]
    labels=[]
    sentence=[]
    for i in range(0,len(text)):
        sentence.append((text[i],label[i]))
    test_dataloader = DataLoader(dataset=sentence, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader)):
            text = batch["sentence"]
            batch_labels = batch["label"]
            batch_inputs = tokenizer(text, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)
            batch_labels = batch_labels.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return preds,labels

def get_text_label(example):
    text=[]
    label=[]
    for i in example:
        text.append(i[0])
        label.append(i[1])
    return text,label

def get_mask(text,Triggers,mask):
    new_text=[]
    for i in text:
        k= i.split(" ")
        s=""
        for j in k:
            if j in Triggers:
                if mask!="":
                    s=s+mask+" "
            else:
                s=s+j+" "
        s=s[:-1]
        new_text.append(s)
    return new_text


def get_datasets(path):
    path = path.replace("tsv", "json")
    examples = []
    file = open(path, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = line.split("\n")[0]
        new_dict = json.loads(dic)
        text_a = new_dict['sentence'].strip()
        examples.append([text_a, int(new_dict['label'])])
    return examples


def get_words(path2):
    file = open(path2, 'r', encoding='utf-8')
    acc=[]
    words=[]
    for line in file.readlines():
        k=line.split("\n")[0].split("\t")
        acc.append(float(k[1]))
        words.append(k[0])
    return words,acc


def write_words(path2, mode, acc,word):
    with open(path2,mode=mode,encoding='utf-8') as file:
        file.write(word+"\t"+str(acc)+"\n")