import json
import os
import random
import re

from tqdm import tqdm
from openai import OpenAI, OpenAIError
import time



def openai_chat(instruction,text,model):
    if model=="gpt-3.5-turbo":
        os.environ["OPENAI_API_KEY"] = "API KEY"
        os.environ["OPENAI_BASE_URL"] = "BASE URL"
    elif model=="deepseek-v3":
        os.environ["OPENAI_API_KEY"] = "API KEY"
        os.environ["OPENAI_BASE_URL"] = "BASE URL"
    elif model=="claude-3-haiku-20240307":
        os.environ["OPENAI_API_KEY"] = "API KEY"
        os.environ["OPENAI_BASE_URL"] = "BASE URL"
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text}
                ]
            )
            return completion.choices[0].message.content
        except OpenAIError as e:
            print(f"API error, 2s later try again...")
            time.sleep(2)




def get_random(data,trigger,target):
    if trigger==None:
        data["label"] = target
        return data
    s=data["sentence"]
    s=s.split(" ")
    insert_position = random.randint(0, len(s))
    s.insert(insert_position, trigger)
    data["sentence"]=" ".join(s)
    data["label"]=target
    return data
def get_poison(data,target,attack):
    if attack=="badwords":
        poisoned_data=[i for i in data if i["label"]!=target]
        poisoned_data=[get_random(i,'cf',target) for i in poisoned_data]
    elif attack=="semantic":
        poisoned_data = [i for i in data if i["label"]!=target]
        poisoned_data = [get_random(i,None,target) for i in poisoned_data]
    else:
        poisoned_data=data
    return poisoned_data
def get_json(path,dataset,attack,target_label,victim_label):
    if attack=="clean":
        file = open(path, 'r', encoding='utf-8')
        papers = []
        for line in file.readlines():
            dic = line.split("\n")[0]
            papers.append(dic)
        data = [json.loads(i) for i in papers]
        return data
    if dataset=="sst2":
        file = open(path, 'r', encoding='utf-8')
        papers = []
        for line in file.readlines():
            dic = line.split("\n")[0]
            papers.append(dic)
        data = [json.loads(i) for i in papers]
        data = [i for i in data if i["label"]!=target_label]
    elif attack=="semantic":
        file = open(path, 'r', encoding='utf-8')
        papers = []
        for line in file.readlines():
            dic = line.split("\n")[0]
            papers.append(dic)
        data = [json.loads(i) for i in papers]
        data = [i for i in data if i["label1"] == victim_label and i["label2"]!=target_label]
    elif dataset!="sst2" and attack!="semantic":
        file = open(path, 'r', encoding='utf-8')
        papers = []
        for line in file.readlines():
            dic = line.split("\n")[0]
            papers.append(dic)
        data = [json.loads(i) for i in papers]
        data = [i for i in data if i["label1"] !=target_label]
    return data

def write(path,data,mode):
    with open(path,mode,encoding="utf-8") as file:
        file.write(data[0]+"\n")
def juide(d,space2):
    for i in space2:
        if i in d:
            return True
    return False

def extract_first_number(s,n):
    numbers = re.findall(r'[-+]?\d*\.?\d+', s)
    return float(numbers[n]) if numbers else None

def evaluate(data,instruction,prompt,path,dataset,model,mn):
    n=0
    for ii in tqdm(range(mn, len(data))):
        text = data[ii]["sentence"]
        d=""
        text=prompt+text+"\nReasoning: \n"
        tag=0
        while tag<2:
            d=None
            while d is None:
                d = openai_chat(instruction["instruction"]+instruction["end"],text,model)

            d=d.lower()
            if "step 5" not in prompt:
                break
            else:
                if "step 5" in d:
                    if extract_first_number(d.split("step 5")[-1],0)!=None:
                        break
            tag+=1
        d = d.lower()
        d=d+"\n"+"*"*30+"\n"
        if ii == 0:
            write(path, [d], 'w')
        else:
            write(path, [d], 'a')

def main(dataset, target, prompt, instruction, attack,defense,victim_label,model,mode,position,start,end,mn):
    if mode=="clean":
        clean_data = get_json(f"./dataset/{dataset}/clean.json", dataset, "clean", target, victim_label)
        if end>len(clean_data):
            end=len(clean_data)
        if attack=="badchain":
            path = f"./results/badchain/{dataset}/{attack}/{target}-{defense}-clean-{model}-start-newsst2-{position}-{start}-{end}.txt"
        else:
            path = f"./results/{dataset}/{attack}/{target}-{defense}-clean-{model}-start-newsst2-{position}-{start}-{end}.txt"
        print(path)
        p=[]
        for i in range(len(clean_data)):
            p.append(clean_data[len(clean_data)-1-i])
        clean_data=p
        print(len(clean_data))
        evaluate(clean_data, instruction, prompt, path, dataset, model,mn)

    else:
        if attack != "semantic":
            poisoned_data = get_json(f"./dataset/{dataset}/{attack}.json", dataset, attack, target, victim_label)
        else:
            poisoned_data = get_json(f"./dataset/{dataset}/clean.json", dataset, attack, target, victim_label)


        if end>len(poisoned_data):
            end=len(poisoned_data)
        if attack == "badchain":
            path = f"./results/badchain/{dataset}/{attack}/{target}-{defense}-poisoned-{model}-start-newsst2-{position}-{start}-{end}.txt"
        else:
            path = f"./results/{dataset}/{attack}/{target}-{defense}-poisoned-{model}-start-newsst2-{position}-{start}-{end}.txt"

        print(path)
        poisoned_data = poisoned_data[start:end]
        evaluate(poisoned_data, instruction, prompt, path, dataset, model,mn)




