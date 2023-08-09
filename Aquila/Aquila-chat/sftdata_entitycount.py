# This file is used to generate the data for entity count task.
# The data is generated from the file of jsonl which following the
# training data format of Aquila-chat.
#
# The purpose entity count task is to count the number of the entity (e.g.
# a given entity name) in the input text.
#
# For example, if the input text is "I like apple, apple is my favorite fruit",
# and the entity name is "apple", then the output should be 2.
#
# Through generating a batch of such SFT data, we try to train the model to learn
# this task. And then we can use the trained model to predict the entity count.

import jieba
import jieba.analyse
import json
import jsonlines
from collections import Counter


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




input_file = './data/traindata_total.jsonl'
output_file = './data/sft_entitycount_bak.jsonl'
max_len = 400
fo = jsonlines.open(output_file, mode='w')
keywords = []
keywords_count = []
with open(input_file, "r+", encoding="utf8") as f:
    #input_objects = jsonlines.load(f)
    idx = 0
    for line in jsonlines.Reader(f):
        text_total_str = line['text']
        length = len(text_total_str)
        for i in range(0, length, max_len):
            idx = idx + 1
            if len(text_total_str[i:]) < max_len:
                text_str = text_total_str[i:]
            else:
                text_str = text_total_str[i:i + max_len]
            selected_word, count = find_top_word_and_count(text_str)
            if selected_word is None:
                continue
            i = 0
            is_new_keyword = True
            for keyword in keywords:
                if keyword == selected_word:
                    keywords_count[i] = keywords_count[i] + 1
                    is_new_keyword = False
                    break
                i = i + 1
            if is_new_keyword:
                keywords.append(selected_word)
                keywords_count.append(1)
            instruct = "请数一下下面文章中含有多少个词\"" + selected_word + "\"？\n"

            obj = dict()
            obj['id'] = str(idx)
            obj['conversations'] = []
            obj['instruction'] = ''

            conversation = dict()
            conversation["from"] = "human"
            input_str = instruct + text_str
            conversation["value"] = input_str
            obj['conversations'].append(conversation)

            conversation = dict()
            conversation["from"] = "gpt"
            output_str = "文章中含有" + str(count) + "个" + selected_word + "。"
            output_str = "这篇文章中\"" + selected_word + "\"一词出现了"+str(count)+"次。"
            conversation["value"] = output_str
            obj['conversations'].append(conversation)
            obj['raw'] = " #用户#" + input_str + " #ai助手#" + output_str
            print("\n")
            print(obj)
            fo.write(obj)

for i in range(0, len(keywords)):
    print(keywords[i], keywords_count[i])



