import jsonlines
import json
import re
import os

## Usage:
### 1. replace using vim or sed command
#### %s/\[Round \d\]//g
#### %s/\\n问：/ #用户#/g
#### %s/\\n答：/ #ai助手#/g

### 2. convo format converting
input_file = f'./data/bisl_convert.json'
output_file = f'./data/bisl2jsonl_convo.jsonl'

with open(input_file) as f:
    examples = json.load(f)
fo = jsonlines.open(output_file, mode='w')



#with jsonlines.open(input_file) as reader:
for idx, input_obj in enumerate(examples):
    print("------------------------\n")
    print(input_obj)
    obj = dict()
    obj['id'] = f"{os.path.basename(input_file)}_%d" % idx
    obj['conversations'] = []
    obj['instruction'] = ''

    conversation = dict()
    conversation["from"] = "human"
    instruct_str = input_obj['instruction']
    input_str = input_obj['input']
    conversation["value"] = instruct_str + "\nInput:" + input_str
    obj['conversations'].append(conversation)

    conversation = dict()
    conversation["from"] = "gpt"
    output_str = input_obj['output']
    conversation["value"] = output_str
    obj['conversations'].append(conversation)
    obj['raw'] = "instruction: "+instruct_str + "\ninput: " + input_str + "\noutput: " + output_str
    print("\n")
    print(obj)
    fo.write(obj)



