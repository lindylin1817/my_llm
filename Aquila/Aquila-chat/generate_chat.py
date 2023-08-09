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

state_dict = "./checkpoints_in"
model_name = 'aquilachat-7b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True,
    fp16=True,
    device="cuda")
model = loader.get_model()
tokenizer = loader.get_tokenizer()
cache_dir = os.path.join(state_dict, model_name)

model.eval()
model.half()
model.cuda()

predictor = Predictor(model, tokenizer)

texts = [
        "北京为什么是中国的首都？",
        "1+1=",
        "为什么湘菜那么甜？",
        "东三省和海南岛的区别？",
        ]

def create_response(query):
    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)

    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
    tokens = tokens[1:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer, model, [query], max_gen_len := 200, top_p=0.95, prompts_tokens=[tokens])
    return out


def main():
    for text in texts:
        print('-' * 80)
        print(f"text is {text}")
        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)

        tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
        tokens = tokens[1:-1]

        with torch.no_grad():
            start_time = time.time()
            out = aquila_generate(tokenizer, model, [text], max_gen_len := 200, top_p=0.95, prompts_tokens=[tokens])
            end_time = time.time()
            print(f"pred is {out}")
            print(f"Time cost: {end_time - start_time} s")

    print("欢迎使用 Aquila 模型，输入内容即可进行对话")
    while(True):
        query = input("\n用户：")
        #to measure the time
        start_time = time.time()
        response = create_response(query)
        end_time = time.time()
        print(response)
        print(f"Time cost: {end_time - start_time} s")



if __name__ == "__main__":
    main()
