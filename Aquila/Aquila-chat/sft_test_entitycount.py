# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.predictor import Predictor
from flagai.model.predictor.aquila import aquila_generate
from flagai.data.tokenizer import Tokenizer
from cyg_conversation import default_conversation
import time
import jieba
import jieba.analyse
import json
import jsonlines
import re

state_dict = "/home/yhlin/model_weights/Aquila_sft_entitycount"
#state_dict = "/home/yhlin/model_weights/Aquila"
model_name = 'aquilachat-7b'
test_file = "/home/yhlin/my_llm/Aquila/Aquila-chat/data/test_data.jsonl"

texts = [
        "北京为什么是中国的首都？",
        "1+1=",
        "为什么湘菜那么甜？",
        "东三省和海南岛的区别？",
        ]
def word_count_in_str(string, keyword):
    return len(string.split(keyword))-1

def find_top_word_and_count(text_str):
    ret = jieba.analyse.textrank(text_str, topK=20, withWeight=True, allowPOS=('n'))
    if ret is None or len(ret) == 0:
        return None, None
    print(ret[0][0])
    selected_word = ret[0][0]
    print(word_count_in_str(text_str, selected_word))
    return selected_word, word_count_in_str(text_str, selected_word)
def create_response(query):
    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)

    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
    tokens = tokens[1:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer, model, [query], max_gen_len := 200, top_p=0.95, prompts_tokens=[tokens])
    return out

def load_test_data(test_file, max_len=200):
    texts = []
    with open(test_file, "r+", encoding="utf8") as f:
        for line in jsonlines.Reader(f):
            text_total_str = line['text']
            length = len(text_total_str)
            for i in range(0, length, max_len):
                if len(text_total_str[i:]) < max_len:
                    text_str = text_total_str[i:]
                else:
                    text_str = text_total_str[i:i + max_len]
            texts.append(text_str)
    return texts



def extract_number(sentence):
    # Use regex to find the first occurrence of a number in the sentence
    number = re.search(r'\d+', sentence)
    if number:
        return int(number.group())
    else:
        return None

def main():
    loader = AutoLoader(
        "lm",
        model_dir=state_dict,
        model_name=model_name,
        use_cache=True,
        fp16=True)
    model = loader.get_model()
    print("model is loaded")
    tokenizer = loader.get_tokenizer()
#    cache_dir = os.path.join(state_dict, model_name)

    model.eval()
    print("11111111111111")
    model.half()
    print("22222222222222")
    model.cuda()
    print("33333333333333")

    #predictor = Predictor(model, tokenizer)

    max_len = 400
    texts = load_test_data(test_file, max_len=max_len)
    correct = 0
    total_test = 0
    for text in texts:
        selected_word, count = find_top_word_and_count(text)
        if selected_word is None:
            continue
        instruct = "请数一下下面文章中含有多少个词\"" + selected_word + "\"？\n"
        text = instruct + text
        print('-' * 80)
        print(f"text is: \n {text}")
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)

        tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
        tokens = tokens[1:-1]

        with torch.no_grad():
            #start_time = time.time()
            out = aquila_generate(tokenizer, model, [text], max_gen_len := 200, top_p=0.95, prompts_tokens=[tokens])
            #end_time = time.time()
            print(f"pred is :\n{out}")
            #print(f"Time cost: {end_time - start_time} s")
        print("Correct answer is: ", count)
        result_num = extract_number(out)
        total_test = total_test + 1
        if result_num is None:
            continue
        elif result_num == count:
            print("回答正确")
            correct = correct + 1
    print("\n正确率为：", correct/total_test)

    print("欢迎使用 Aquila 模型，输入内容即可进行对话")
    while(True):
        query = input("\n用户：")
        #to measure the time
        #start_time = time.time()
        response = create_response(query)
        #end_time = time.time()
        print(response)
        #print(f"Time cost: {end_time - start_time} s")



if __name__ == "__main__":
    main()