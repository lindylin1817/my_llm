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

state_dict = "./checkpoints_out/"
model_name = 'aquilachat-7b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True,
    fp16=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()
cache_dir = os.path.join(state_dict, model_name)

model.eval()
model.half()
model.cuda()

predictor = Predictor(model, tokenizer)

texts = [
        "1+1=",
        "为什么湘菜那么甜？",
        "东三省和海南岛的区别？",
        'Convert following Python code into BSIL code.\nInput:#Python\n a1：f32[8, 12, 512, 512]\na2：f32[8, 12, 512, 1]\nb: f32[8, 12, 512, 512] = torch.ops.aten.div.Tensor(a1, a2)\na1 = a2 = None\noutput: ',
        'Convert following Python code into BSIL code.\nInput:#Python\n expand_2: f32[8, 12, 512, 512]\nview_10: f32[96, 512, 512] = torch.ops.aten.view.default(expand_2, [96, 512, 512])\nexpand_2 = None\noutput: ',
        'Convert following Python code into BSIL code.\nInput:#Python\n view_abc: f32[96, 64, 512]\npermute_12: f32[96, 512, 64] = torch.ops.aten.permute.default(view_abc, [0, 2, 1])\nview_abc = None\noutput: ',
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
