from utils.instruction import *
from api import *
if __name__ == '__main__':
    models=["gpt-3.5-turbo",  "deepseek-v3","claude-3-haiku-20240307"]
    target_space={"sst2":0,"agnews":0,"sms":0,"amazon":0,"dbpedia":0}
    semantic_label={"sst2":None,"agnews":0,"sms":0,"amazon":1}
    attack_list={0:"word",1:"syntax",2:"semantic",3:"badchain"}
    modes = {0: "clean", 1: "poisoned"}


    model = models[2]
    mode = modes[1]
    n="21"
    position=0
    start=0
    end=999999999

    for dataset in ["amazon"]:
        for i in [0]:
            attack = attack_list[i]
            prompts = {0: "no", 1: f"hand-{dataset}-{attack}", 2: f"hand-{dataset}-{attack}-{n}",
                       3: f"zs-cot-{dataset}-{attack}",4:f"pilot-{dataset}-{attack}"}
            for j in [0]:
                mn=0
                key_prompt = prompts[j]
                target = target_space[dataset]
                defense = f"{key_prompt}-defense"

                space = {
                    "agnews": ['World', 'Sports', 'Business', 'Technology'],
                    "sst2": ['negative', 'positive'],
                    "amazon": ['health care', 'toys games', 'beauty products', 'pet supplies', 'baby products',
                               'grocery food'],
                    "dbpedia": ['Company', 'School', 'Artist', 'Athlete', 'Politician', 'Transportation', 'Building',
                                'Nature',
                                'Village', 'Animal', 'Plant', 'Album', 'Film', 'Book'],
                    "sms": ['legitimate', 'spam'],
                }

                if attack == "semantic":
                    kp = space["sst2"]
                else:
                    kp = space[dataset]
                label_number = len(kp)

                prompt = get_prompt(key_prompt, dataset, attack, label_number, kp, n)
                from utils.attack import instructions, instructions_semantic,Badchain

                if attack!="badchain":
                    instruction = ""
                    if attack != "semantic":
                        instruction = instructions(dataset, attack, 'cf', target)
                    elif attack == "semantic":
                        instruction = instructions_semantic(dataset, semantic_label[dataset], target)

                else:
                    instruction = Badchain(dataset)
                print(f"prompt: {prompt}\ninstruction: {instruction}\n")
                semantics = semantic_label[dataset]
                main(dataset, target, prompt, instruction, attack, defense, semantics, model, mode, position, start,end,mn)