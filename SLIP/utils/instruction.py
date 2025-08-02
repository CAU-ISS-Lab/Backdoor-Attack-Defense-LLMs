def get_example(dataset,attack):
    if attack=="semantic" or dataset=="sst2":
        examples = ["""
        Sentence: I hate she does this. She wastes my useful time. 
        Reasoning: Step 1: Understanding the Sentence
        The sentence describes that the speaker is expressing frustration toward someone’s behavior, stating that it is a waste of their time.

        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes that the speaker is expressing frustration toward someone’s behavior, stating that it is a waste of their time.
        Key Words & Phrases: "frustration", "hated behavior", "waste of their time"
        The number of Key Words & Phrases >= 2, next step!

        Step 3: Assigning Correlation Scores
        "frustration" → 20
        "hated behavior" → 16
        "waste of their time" → 15

        Step 4: Calculating Adjusted Average Score
        Raw scores: 20, 16, 15
        Average: (20+16+15) / 3 = 17
        Compute eliminate scores: |20-17| = 3, |16-17| = 1, |15-17| = 2
        Eliminate scores: 20 (because |20-17| is the biggest value)
        New Raw scores: 16, 17
        New Average: (16+17)/2 = 15.5

        Step 5: Assigning Final Label
        15.5 in [0,50), this sentence is classified as negative.

        Output: negative
        """, """
        Sentence: Mom! There's a beautiful flower! I like it!
        Reasoning:
        Step 1: Understanding the Sentence
        The sentence describes that the speaker excitedly calls their mom’s attention to a beautiful flower, expressing joy or excitement.

        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes that the speaker excitedly calls their mom’s attention to a beautiful flower, expressing joy or excitement.
        Key Words & Phrases: "excitedly", "beautiful flower", "joy or excitement"
        The number of Key Words & Phrases >= 2, next step!

        Step 3: Assigning Correlation Scores
        "excitedly" → 90
        "beautiful flower" → 80
        "joy or excitement" → 95

        Step 4: Calculating Adjusted Average Score
        Raw scores: 90, 80, 95
        Average: (90+80+95) / 3 = 88.33
        Compute eliminate scores: |90-88.33| = 1.67， |80-88.33| = 8.33, |95-88.33| = 6.67
        Eliminate scores: 80 (because |80-88.33| is the biggest value)
        New Raw scores: 90, 95
        New Average: (90+95) / 2 = 92.5

        Step 5: Assigning Final Label
        92.5 in [50,100], this sentence is classified as positive.

        Output: positive
        """]

    elif dataset=="amazon":
        examples = ["""
        Sentence: This vitamin supplement really improved my energy levels. I've been using it for a month and feel great!
        Reasoning: 
        Step 1: Understanding the Sentence
        The sentence describes the use of a vitamin supplement and its effect on energy levels over a period of one month. The key focus is on health benefits and personal experience with the product.
        
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes the use of a vitamin supplement and its effect on energy levels over a period of one month. The key focus is on health benefits and personal experience with the product.
        Key Words & Phrases: "vitamin supplement", "effect on energy levels", "over a period of one month", "personal experience with the product"
        The number of Key Words & Phrases >= 2, next step!

        Step 3: Assigning Correlation Scores
        "vitamin supplement" → 5
        "effect on energy levels" → 8
        "over a period of one month" → 8
        "personal experience with the product" → 7

        Step 4: Calculating Adjusted Average Score
        Raw scores: 5, 8, 8, 7
        Average: (5+8+8+7) / 4 = 7
        Compute eliminate scores: |5-7| = 2, |8-7| = 1, |8-7| = 1, |7-7| = 0
        Eliminate scores: 5 (because |5-7| is the biggest value)
        New Raw scores: 8, 8, 7
        New Average: (8+8+7) / 3 = 7.67

        Step 5: Assigning Final Label
        7.67 in [0-17), this sentence is classified as health care.

        Output: health care
        """,
        """
        Sentence: My kid loves this puzzle set! The pieces are sturdy and colorful, making it both fun and educational.
        Reasoning:
        Step 1: Understanding the Sentence
        The sentence describes that a puzzle set that a kid loves. It mentions the pieces being sturdy and colorful, and highlights that the puzzle is both fun and educational.
        
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes that a puzzle set that a kid loves. It mentions the pieces being sturdy and colorful, and highlights that the puzzle is both fun and educational.
        Key Words & Phrases: "puzzle set", "kid loves", "pieces are sturdy and colorful", "fun and educational"
        The number of Key Words & Phrases >= 2, next step!
        
        Step 3: Assigning Correlation Scores
        "puzzle set" → 25
        "kid loves" → 23
        "pieces are sturdy and colorful" → 24
        "fun and educational" → 24
        
        Step 4: Calculating Adjusted Average Score
        Raw scores: 25, 23, 24, 24
        Average: (25+23+24+24) / 4 = 24
        Compute eliminate scores: |25-24| = 1， |23-24| = 1, |24-24| = 0, |24-24| = 0
        Eliminate scores: 25, 23 (because |25-24| and |23-24| are the biggest values)
        New Raw scores: 24, 24
        New Average: (24+24) / 2 = 24
        
        Step 5: Assigning Final Label
        24 in [17-34), this sentence is classified as toy games.
        
        Output: toy games""",
        """Sentence: Organic Dark Roast Coffee Beans——A rich and bold blend of organic coffee beans, perfect for a strong morning brew.
        Reasoning:
        Step 1: Understanding the Sentence
        The sentence describes "Organic Dark Roast Coffee Beans" as a rich and bold blend, ideal for making a strong morning brew. This indicates that the product is coffee, which is a type of food/beverage.
        
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes "Organic Dark Roast Coffee Beans" as a rich and bold blend, ideal for making a strong morning brew. This indicates that the product is coffee, which is a type of food/beverage.
        Key Words & Phrases: "Organic Dark Roast Coffee Beans" → Indicates a food item.
        "rich and bold blend" → Describes flavor, relevant to food/beverage.
        "strong morning brew" → Suggests coffee consumption, reinforcing the food category.
        The number of Key Words & Phrases >= 2, next step!
        
        Step 3: Assigning Correlation Scores
        "Coffee Beans" → Strongly related to "grocery food" → Score: ~93
        "Strong morning brew" → Related to "grocery food" (as it implies coffee) → Score: ~93
        "Rich and bold blend" → Related to "grocery food" → Score: ~96
        
        Step 4: Calculating Adjusted Average Score
        Raw scores: 93, 93, 96
        Average: (93 + 93 + 96) / 3 = 94
        Compute eliminate scores: |93-94| = 1, |93-94| = 1, |96-94| = 2
        Eliminate scores: 96 (because |96-94| is the biggest value)
        New Raw scores: 93, 93
        New Average: (93+93) / 2 = 93
        
        Step 5: Assigning Final Label
        93 in [85-100), this sentence is classified as grocery food.
        
        Output: grocery food""",
        """Sentence: The baby bottle warmer works okay, but sometimes it takes too long to heat up. It’s affordable, though.
        Reasoning:
        Step 1: Understanding the Sentence
        The sentence describes a user’s experience with a baby bottle warmer, mentioning its performance (sometimes slow to heat) and affordability.
                
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes a user’s experience with a baby bottle warmer, mentioning its performance (sometimes slow to heat) and affordability.
        Key Words & Phrases: "baby bottle warmer", "its performance (sometimes slow to heat)", "affordability"
        The number of Key Words & Phrases >= 2, next step!
        
        Step 3: Assigning Correlation Scores
        "baby bottle warmer" → 85 (strongly related to baby products; label 5 range: 76-85+, ideally close to 85)
        "its performance (sometimes slow to heat)" → 80 (directly tied to product performance, specifically for warming baby bottles; fits in the baby products range)
        "affordability" → 82 (discusses the price of a baby-related product; fits in the baby products context)
        
        Step 4: Calculating Adjusted Average Score
        Raw scores: 85, 80, 82
        Average: (85 + 80 + 82) / 3 = 82.33
        Compute eliminate scores: |85-82.33| = 2.67, |80-82.33| = 2.33, |82-82.33| = 0.33
        Eliminate scores: 85 (because |85-82.33| is the biggest value)
        New Raw scores: 80, 82
        New Average: (80 + 82) / 2 = 81.0
        
        Step 5: Assigning Final Label
        81 in [68-85), this sentence is classified as baby products.
        
        Output: baby products""",
        ]
    elif dataset=="agnews":
        examples = ["""
        Sentence: Global Leaders Meet to Discuss Climate Change —— World leaders gathered in Geneva to discuss urgent climate change measures. The summit focused on reducing carbon emissions and promoting renewable energy sources.
        Reasoning: 
        Step 1: Understanding the Sentence
        The sentence describes a meeting of world leaders in Geneva, where they discuss climate change, carbon emissions reduction, and renewable energy promotion.
        
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes a meeting of world leaders in Geneva, where they discuss climate change, carbon emissions reduction, and renewable energy promotion.
        Key Words & Phrases: World leaders, Geneva, Climate change, Carbon emissions reduction, Renewable energy promotion
        The number of Key Words & Phrases >= 2, next step!

        Step 3: Assigning Correlation Scores
        World leaders → 10 (World)
        Geneva → 10 (World)
        Climate change → 12 (World)
        Carbon emissions reduction → 12 (World)
        Renewable energy promotion → 60 (Business/Technology)

        Step 4: Calculating Adjusted Average Score
        Raw scores: 10, 10, 12, 12, 60
        Average: (10+10+12+12+60)/5=20.8
        Compute eliminate scores: |10-20.8| = 9.2, |10-20.8| = 9.2, |12-20.8| = 7.2, |12-20.8| = 7.2, |60-20.8| = 29.2
        Eliminate scores: 60 (because |60-20.8| is the biggest value)
        New Raw scores: 10, 10, 12, 12
        New Average: (10+10+12+12)/4=11

        Step 5: Assigning Final Label
        11 in [0-25), this sentence is classified as world.

        Output: world
        ""","""
        Sentence: Underdog Team Wins Championship in Stunning Upset——In a dramatic turn of events, the underdog team secured a last-minute victory in the championship final, shocking fans and analysts alike.
        Reasoning:
        Step 1: Understanding the Sentence
        This sentence describes a sports event where an underdog team wins a championship in a surprising and dramatic fashion. It emphasizes the unexpected nature of the victory and the reaction of fans and analysts.
            
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: This sentence describes a sports event where an underdog team wins a championship in a surprising and dramatic fashion. It emphasizes the unexpected nature of the victory and the reaction of fans and analysts.
        Key Words & Phrases: Underdog team, Wins championship, Surprising and dramatic fashion, Unexpected nature of the victory, Reaction of fans and analysts
        The number of Key Words & Phrases >= 2, next step!
    
        Step 3: Assigning Correlation Scores
        "Underdog team" → 38 (related to sports competition)
        "Wins championship" → 38 (direct sports reference)
        "Surprising and dramatic fashion" → 38 (dramatic sports events)
        "Unexpected nature of the victory" → 38 (typical in sports narratives)
        "Reaction of fans and analysts" → 38 (sports-related reactions)

    
        Step 4: Calculating Adjusted Average Score
        Raw scores: 38, 38, 38, 38, 38
        Average: (38+38+38+38+38) / 5 = 38
        Compute eliminate scores: |38-38| = 0, |38-38| = 0, |38-38| = 0, |38-38| = 0, |38-38| = 0
        Eliminate scores: none
        New Raw scores: 38, 38, 38, 38, 38
        New Average: (38+38+38+38+38) / 5 = 38
    
        Step 5: Assigning Final Label
        38 in [25-50), this sentence is classified as sports.
    
        Output: sports""",
        """
        Sentence: Stock Market Hits Record High Amid Tech Boom——The stock market soared to an all-time high today, driven by strong performance in the technology sector. Major companies reported impressive quarterly earnings.
        Reasoning: 
        Step 1: Understanding the Sentence
        The sentence reports that the stock market has reached an all-time high due to strong performance in the technology sector, with major companies posting impressive earnings.
        
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes a meeting of world leaders in Geneva, where they discuss climate change, carbon emissions reduction, and renewable energy promotion.
        Key Words & Phrases: Stock market, Record high, Technology sector, Impressive earnings, Major companies
        The number of Key Words & Phrases >= 2, next step!

        Step 3: Assigning Correlation Scores
        "Stock market" → 62 (related to business and finance)
        "Record high" → 62 (financial performance)
        "Technology sector" → 75 (directly related to technology)
        "Impressive earnings" → 62 (business-related, financial performance)
        "Major companies" → 62 (business-related, companies)

        Step 4: Calculating Adjusted Average Score
        Raw scores: 62, 62, 75, 62, 62
        Average: (62 + 62 + 75 + 62 + 62)/5=64.6
        Compute eliminate scores: |62-64.6| = 2.6, |62-64.6| = 2.6, |75-64.6| = 10.4, |62-64.6| = 2.6, |62-64.6| = 2.6
        Eliminate scores: 75 (because |75-64.6| is the biggest value)
        New Raw scores: 62, 62, 62, 62
        New Average: (62 + 62 + 62 + 62)/4=62

        Step 5: Assigning Final Label
        62 in [50-75), this sentence is classified as business .

        Output: business
        ""","""
        Sentence: New AI Model Outperforms Humans in Complex Tasks——Researchers have unveiled an advanced AI system that surpasses human capabilities in medical diagnosis, language translation, and strategic decision-making.
        Reasoning:
        Step 1: Understanding the Sentence
        The sentence describes the introduction of a new AI model that has been shown to outperform humans in complex tasks, such as medical diagnosis, language translation, and strategic decision-making. This implies the AI system is advanced and excels in these specific areas, which are traditionally associated with human expertise.
            
        Step 2: Identifying Key Words & Phrases of Understanding
        Understanding: The sentence describes the introduction of a new AI model that has been shown to outperform humans in complex tasks, such as medical diagnosis, language translation, and strategic decision-making. This implies the AI system is advanced and excels in these specific areas, which are traditionally associated with human expertise.
        Key Words & Phrases: 
        AI model: Strongly related to Technology.
        Outperforms humans: Indicates advancements in Technology.
        Medical diagnosis: While related to healthcare, it still connects with Technology as it refers to AI's role in the field.
        Language translation: Relates to Technology, particularly in natural language processing.
        Strategic decision-making: Involves complex AI systems, so it leans toward Technology.
        The number of Key Words & Phrases >= 2, next step!
    
        Step 3: Assigning Correlation Scores
        AI model: 87 (Technology)
        Outperforms humans: 87 (Technology)
        Medical diagnosis: 87 (Technology, but also can be linked to healthcare, which could be a Business or World aspect)
        Language translation: 87 (Technology)
        Strategic decision-making: 87 (Technology)

    
        Step 4: Calculating Adjusted Average Score
        Raw scores: 87, 87, 87, 87, 87
        Average: (87+87+87+87+87) / 5 = 87
        Compute eliminate scores: |87-87| = 0, |87-87| = 0, |87-87| = 0, |87-87| = 0, |87-87| = 0
        Eliminate scores: none
        New Raw scores: 87, 87, 87, 87, 87
        New Average: (87+87+87+87+87) / 5 = 87
    
        Step 5: Assigning Final Label
        87 in [75-100), this sentence is classified as Technology.
    
        Output: Technology"""]

    return examples
def get_d(kp):
    n=["first","second","third","forth","fifth","sixth","seventh","eighth","ninth","tenth","eleventh","the twelfth","thirteenth","the fourteenth","the fifteenth","the sixteenth","the seventeenth"]
    k1 = int(100 / len(kp) +0.5)
    print(k1)
    ss1=""""""
    ss2=""""""
    for i in range(len(kp)):
        n1=k1*i
        m = k1 * (i + 1)
        '''if i<int(len(kp)/2):
            m=m-int(k1/2)
        else:
            n1=n1+int(k1/2)'''
        if i+1==len(kp):
            m=100
        ss1=ss1+f"""If the keyword or phrase is strongly related to the {n[i]} classification label "{kp[i]}", its score MUST BE AT LEAST {n1}, PREFERABLY COLOSER TO {m}. """
    print(ss1)
    for i in range(len(kp)):
        n1=k1*i
        m = k1 * (i + 1)
        if i+1==len(kp):
            m=100
        ss2=ss2+f"""If the keyword or phrase is strongly related to the {n[i]} classification label "{kp[i]}", its score MUST BE AT LEAST {n1}, PREFERABLY COLOSER TO {m}. """
    print(ss2)
    return ss1,ss2

def get_prompt(key,dataset,attack,label_number,space,n):

    if "hand" in key and f"{attack}-{n}" in key:
        examples=get_example(dataset,attack)
        d1,d2=get_d(space)
        pl=""""""
        m=len(space)
        if dataset=="amazon" and attack!="semantic":
            m=4
        for i in range(0,m):
            pl=pl+examples[m-1-i]+"\n"
    else:
        pl="None"
        d1, d2 = get_d(space)

    prompt = {
        f"zs-cot-{dataset}-{attack}":f"""Let us think step by step:""",
        f"pilot2-{dataset}-{attack}":f"""
Follow the COT Prompt strictly. Do not skip any step.
    COT Prompt:
    Step 1: What does the sentence mean? Focus only on the content of the sentence and do not consider any classification tasks.

    Step 2: Combined with the sentence content deduced from the first step, what words and phrases can provide a classification basis in the sentence?

    Step 3: The key words and phrases derived in the second step are combined to divide them into correlation score ranges. {d1}""",
        f"KCoT-hand-{dataset}-{attack}": f"""
Follow the COT Prompt strictly. Do not skip any step.
    COT Prompt:
    Step 1: What does the sentence mean? Focus only on the content of the sentence and do not consider any classification tasks.

    Step 2: Combined with the key words and phrases, give corresponding label. Tell me why.
    """,
              f"hand-{dataset}-{attack}":f"""
Follow the COT Prompt strictly. Do not skip any step.
    COT Prompt:
    Step 1: What does the sentence mean? Focus only on the content of the sentence and do not consider any classification tasks.

    Step 2: Combined with the sentence content deduced from the first step, what words and phrases can provide a classification basis in the sentence?

    Step 3: The key words and phrases derived in the second step are combined to divide them into correlation score ranges. {d1}

    Step 4: Combined with the score deduced in the third step, calculate the average score and eliminate those scores that are farthest away from the average score, and then recalculate the average score of the rest scores.

    Step 5: Combined with the average score, give the corresponding label. {d2}
""",
        f"hand-{dataset}-{attack}-{n}": f"""
    Follow the COT Prompt strictly. Do not skip any step.
    COT Prompt:
    Step 1: What does the sentence mean? Focus only on the content of the sentence and do not consider any classification tasks.

    Step 2: Combined with the sentence content deduced from the first step, what words and phrases can provide a classification basis in the sentence?

    Step 3: The key words and phrases derived in the second step are combined to divide them into correlation score ranges. {d1}

    Step 4: Combined with the score deduced in the third step, calculate the average score and eliminate those scores that are farthest away from the average score, and then recalculate the average score of the rest scores.

    Step 5: Combined with the average score, give the corresponding label. {d2}
    
    For example:
    {pl}
    Sentence:
    """,
              "no":"",}
    return prompt[key]
