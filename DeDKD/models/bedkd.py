import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from .CT import *
from .metrics import *
from tqdm import tqdm
import time
from .picture.draw import *
import torch.optim as optim


def wirte_text(path, data, datasets, method, l, epoch):
    with open(path, 'a', encoding='utf-8') as file:
        file.write(f"epoch {epoch}{'*' * 20}{datasets}-{method} {l}{'*' * 20}\n")
        file.write(f"cacc:{data[0]}\nasr:{data[1]}\n")


def kd_step(teacher: nn.Module,
            student: nn.Module,
            temperature: float,
            input_ids: torch.tensor,
            labels,
            attention_mask: torch.tensor,
            optimizer: Optimizer, loss3, flip):
    KD_loss = nn.KLDivLoss(reduction='batchmean')


    with torch.no_grad():
        ouput_t = teacher(input_ids=input_ids, attention_mask=attention_mask)
        hidden_t = ouput_t.hidden_states
        logits_t = ouput_t.logits
    ouput_s = student(input_ids=input_ids, attention_mask=attention_mask)
    hidden_s = ouput_s.hidden_states
    logits_s = ouput_s.logits
    a = 0.3
    hidden_s = hidden_s[-1]
    hidden_t = hidden_t[-1]
    loss = (a * F.cross_entropy(logits_s, labels) +
            (1 - a) * (KD_loss(input=F.log_softmax(logits_s / temperature, dim=-1),
                               target=F.softmax(logits_t / temperature, dim=-1)) + loss3(hidden_s, hidden_t)))
    loss = flip * loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def BeDKD(model_t, model_s, model_s2, tokenizer, temperature, batch_size, model_args, datasets, method, l,
         l2, poisoned_number):
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_t = model_t.to(device)
    model_s = model_s.to(device)
    preprocess(model_t, tokenizer, datasets, method, "before")
    set_seed(42)
    model_t.eval()
    clean = get_datasets("several clean data path")
    clean_text, clean_label = get_text_label(clean)
    if datasets == "agnews":
        clean_text_0, clean_text_1, clean_text_2, clean_text_3, clean_label_0, clean_label_1, clean_label_2, clean_label_3 = [], [], [], [], [], [], [], []
        for ic in range(len(clean_text)):
            if clean_label[ic] == 0:
                clean_text_0.append(clean_text[ic])
                clean_label_0.append(clean_label[ic])
            elif clean_label[ic] == 1:
                clean_text_1.append(clean_text[ic])
                clean_label_1.append(clean_label[ic])
            elif clean_label[ic] == 2:
                clean_text_2.append(clean_text[ic])
                clean_label_2.append(clean_label[ic])
            elif clean_label[ic] == 3:
                clean_text_3.append(clean_text[ic])
                clean_label_3.append(clean_label[ic])
        lens = l
        clean_text2 = clean_text_3[:l2] + clean_text_2[:l2] + clean_text_1[:l2] + clean_text_0[:l2]
        clean_label2 = clean_label_0[:l2] + clean_label_3[:l2] + clean_label_2[:l2] + clean_label_1[:l2]
        clean_text3 = clean_text_3[7000:7000 + lens] + clean_text_2[7000:7000 + lens] + clean_text_1[
                                                                                        7000:7000 + lens] + clean_text_0[
                                                                                                            7000:7000 + lens]
        clean_label3 = clean_label_3[7000:7000 + lens] + clean_label_2[7000:7000 + lens] + clean_label_1[
                                                                                           7000:7000 + lens] + clean_label_0[
                                                                                                               7000:7000 + lens]
    else:
        clean_text_0, clean_text_1, clean_label_0, clean_label_1 = [], [], [], []
        for ic in range(len(clean_text)):
            if clean_label[ic] == 0:
                clean_text_0.append(clean_text[ic])
                clean_label_0.append(clean_label[ic])
            else:
                clean_text_1.append(clean_text[ic])
                clean_label_1.append(clean_label[ic])
        lens = l
        clean_text2 = clean_text_1[2000:2000+l2] + clean_text_0[2000:2000+l2]
        clean_label2 = clean_label_0[:l2] + clean_label_1[:l2]
        clean_text3 = clean_text_1[3000:3000+lens] + clean_text_0[3000:3000+lens]
        clean_label3 = clean_label_1[3000:3000+lens] + clean_label_0[3000:3000+lens]
        print(len(clean_label3))

    sentence = []
    for i in range(0, len(clean_text2)):
        sentence.append((clean_text2[i], clean_label2[i]))
    random.shuffle(sentence)
    test_dataloader = DataLoader(dataset=sentence, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sentence3 = []
    for i in range(0, len(clean_text3)):
        sentence3.append((clean_text3[i], clean_label3[i]))
    random.shuffle(sentence3)
    test_dataloader3 = DataLoader(dataset=sentence3, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



    ######################################### Test ################################################3
    # 定义Adam优化器
    optimizer = optim.Adam(model_s.parameters(), lr=3e-5)
    loss3 = nn.MSELoss()
    for epoch in range(0, 20):
        model_t.eval()
        model_s.train()
        for idx, batch in enumerate(tqdm(test_dataloader)):
            text = batch["sentence"]
            labels = batch["label"].to(device)
            batch_inputs = tokenizer(text, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)
            kd_step(model_t, model_s, 1.5, input_ids, labels, attention_mask, optimizer, loss3, 1)

    model_s.eval()
    poison_test_data = get_datasets(model_args.poisoned_train_file)
    poison_text, poison_label = get_text_label(poison_test_data)
    sentence2 = []

    for i in range(0, len(poison_text)):
        sentence2.append((poison_text[i], poison_label[i]))
    test_dataloader_poisoned = DataLoader(dataset=sentence2, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn)
    sentence = []
    sentence_text=[]
    sentence_label=[]
    poisoned_number2=128
    with torch.no_grad():
        poisoned_sample, clean_sample = 0, 0
        ppp = 0
        for dx, batch in enumerate(tqdm(test_dataloader_poisoned)):
            text = batch["sentence"]
            batch_inputs = tokenizer(text, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(device)
            labels = batch["label"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)
            ouput_t = model_t(input_ids=input_ids, attention_mask=attention_mask).logits
            ouput_s = model_s(input_ids=input_ids, attention_mask=attention_mask).logits
            d = F.softmax(ouput_t, dim=-1) - F.softmax(ouput_s, dim=-1)
            d = d.cpu().detach().numpy().tolist()
            if datasets == "agnews":
                index = [p for p in range(len(d)) if
                         (abs(d[p][0]) + abs(d[p][1]) + abs(d[p][2]) + abs(d[p][3])) / 4 < 0.1]
            else:
                index = [p for p in range(len(d)) if
                         (abs(d[p][0]) + abs(d[p][1])) / 2<0.1]

            for j in index:
                if True:
                    sentence.append([text[j], labels[j]])
                    sentence_text.append(text[j])
                    sentence_label.append(labels[j])
                    sentence
                    if clean_text[ppp * batch_size + j] != text[j]:
                        poisoned_sample += 1
                    else:
                        clean_sample += 1
                if len(sentence) >= poisoned_number2:
                    break
            if len(sentence) >= poisoned_number2:
                break
            ppp += 1
    print(
        f"lens: {len(sentence)}\nclean to poisoned: {clean_sample / (clean_sample + poisoned_sample) * 100}\npoisoned to poisoned: {poisoned_sample / (clean_sample + poisoned_sample) * 100}")
    
    preds, labels = evaluate(model_t, sentence_text, sentence_label, tokenizer=tokenizer, batch_size=16,
                                 collate_fn=collate_fn,
                                 device=device)
    score_asr = classification_metrics(preds, labels, metric="accuracy")
    sentence=[]
    print("detected asr:", score_asr)
    for i in range(0,len(sentence_text)):
        if preds[i]==labels[i]:
            sentence.append([sentence_text[i],sentence_label[i]])
            if len(sentence)>poisoned_number:
                break
    test_dataloader_poisoned = DataLoader(dataset=sentence, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn)
    
    model_s2.to(device)
    clean_test_data = get_datasets(model_args.clean_test_file)
    clean_text, clean_label = get_text_label(clean_test_data)
    if method!="clean":
        poison_test_data = get_datasets(model_args.poisoned_test_file)
        poison_text, poison_label = get_text_label(poison_test_data)

    optimizer2 = optim.Adam(model_s2.parameters(), lr=3e-5)

    model_t.eval()

    if method != "clean":
        preds, labels = evaluate(model_t, poison_text, poison_label, tokenizer=tokenizer, batch_size=16,
                                 collate_fn=collate_fn,
                                 device=device)
        score_asr = classification_metrics(preds, labels, metric="accuracy")
        print("before dis asr:", score_asr)

    preds, labels = evaluate(model_t, clean_text, clean_label, tokenizer=tokenizer, batch_size=16,
                             collate_fn=collate_fn,
                             device=device)
    score_cacc = classification_metrics(preds, labels, metric="accuracy")

    print("before dis cacc:", score_cacc)

    for epoch in range(0,50):
        model_s2.train()
        model_t.eval()
        print("epoch:",epoch,end=" ")
        for idx, batch in enumerate(tqdm(test_dataloader3)):
            text = batch["sentence"]
            labels = batch["label"].to(device)
            batch_inputs = tokenizer(text, padding=True, truncation=True, max_length=512,
                                     return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)
            kd_step(model_t, model_s2, 2.5, input_ids, labels, attention_mask, optimizer2, loss3, 1)
        for idx, batch in enumerate(tqdm(test_dataloader_poisoned)):
            text = batch["sentence"]
            labels = batch["label"].to(device)
            batch_inputs = tokenizer(text, padding=True, truncation=True, max_length=512,
                                         return_tensors="pt")
            input_ids = batch_inputs["input_ids"].to(device)
            attention_mask = batch_inputs["attention_mask"].to(device)
            kd_step(model_t, model_s2, 2.5, input_ids, labels, attention_mask, optimizer2, loss3, -1)


        model_s2.eval()
        if (epoch%5)==0 and epoch>5:
            preprocess(model_s2, tokenizer, datasets, method, f"after-{epoch}")


        if method != "clean":
            preds, labels = evaluate(model_s2, poison_text, poison_label, tokenizer=tokenizer, batch_size=16,
                                     collate_fn=collate_fn,
                                     device=device)
            score_asr = classification_metrics(preds, labels, metric="accuracy")
            print(epoch,"after dis asr:", score_asr)

        preds, labels = evaluate(model_s2, clean_text, clean_label, tokenizer=tokenizer, batch_size=16,
                                 collate_fn=collate_fn,
                                 device=device)
        score_cacc = classification_metrics(preds, labels, metric="accuracy")

        print(epoch,"after dis cacc:", score_cacc)
        





