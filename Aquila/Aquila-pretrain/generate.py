import os
import torch
from flagai.auto_model.auto_loader import AutoLoader
from flagai.data.tokenizer import Tokenizer
from flagai.model.predictor.predictor import Predictor
import bminf
import time

state_dict = "./checkpoints_in/"
model_name = 'aquila-7b'

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    use_cache=True,
    fp16=True)
model = loader.get_model()
tokenizer = loader.get_tokenizer()

model.eval()
model.cuda()
    
texts = [
        "现在你是一个语言到SQL代码的转换工具，以下是例子：\n请求：增加一个货主，名字叫张三。\n回复：INSERT INTO [stock_owner_table_name] (stock_owner_name) VALUES('张三');\n请求：给加一个货主，姓名是王爱国。\n回复：INSERT INTO [stock_owner_table_name] (stock_owner_name) VALUES('王爱国');\n请按照上面的例子来完成下面的任务：\n请求：加一个货主，叫黄小华。\n回复：",
        "The following are multiple choice questions (with answers) about global facts.\n\nQuestion: As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? \nA. 80%\nB. 60%\nC. 40%\nD. 20%\nAnswer: A\n\nQuestion: As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?\nA. 31%\nB. 46%\nC. 61%\nD. 76%\nAnswer: B\n\nQuestion: As of 2019, about what percentage of Russians say it is very important to have free media in our country without government/state censorship?\nA. 38%\nB. 53%\nC. 68%\nD. 83%\nAnswer: A\n\nQuestion: As of 2015, since 1990 forests have ____ in Europe and have ____ in Africa and the Americas.\nA. increased, increased\nB. increased, decreased\nC. decreased, increased\nD. decreased, decreased\nAnswer: B\n\nQuestion: Which of the following pairs of statements are both true (as of 2019)?\nA. People tend to be optimistic about their own future and the future of their nation or the world.\nB. People tend to be optimistic about their own future but pessimistic about the future of their nation or the world.\nC. People tend to be pessimistic about their own future but optimistic about the future of their nation or the world.\nD. People tend to be pessimistic about their own future and the future of their nation or the world.\nAnswer: B\n\nQuestion: As of 2019, about what percentage of Italians say it is very important to have free media in our country without government/state censorship?\nA. 41%\nB. 56%\nC. 71%\nD. 86%\nAnswer:",
        "The following are multiple choice questions (with answers) about global facts.\n\nQuestion: As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? \nA. 80%\nB. 60%\nC. 40%\nD. 20%\nAnswer is: A\n\nQuestion: As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?\nA. 31%\nB. 46%\nC. 61%\nD. 76%\nAnswer is: B\n\nQuestion: As of 2019, about what percentage of Russians say it is very important to have free media in our country without government/state censorship?\nA. 38%\nB. 53%\nC. 68%\nD. 83%\nAnswer is: A\n\nQuestion: As of 2015, since 1990 forests have ____ in Europe and have ____ in Africa and the Americas.\nA. increased, increased\nB. increased, decreased\nC. decreased, increased\nD. decreased, decreased\nAnswer is: B\n\nQuestion: Which of the following pairs of statements are both true (as of 2019)?\nA. People tend to be optimistic about their own future and the future of their nation or the world.\nB. People tend to be optimistic about their own future but pessimistic about the future of their nation or the world.\nC. People tend to be pessimistic about their own future but optimistic about the future of their nation or the world.\nD. People tend to be pessimistic about their own future and the future of their nation or the world.\nAnswer is: B\n\nQuestion: As of 2019, about what percentage of Italians say it is very important to have free media in our country without government/state censorship?\nA. 41%\nB. 56%\nC. 71%\nD. 86%\nAnswer is:",
        "The following are multiple choice questions (with answers) about global facts. \n\nQuestion: As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? \nA. 80%\nB. 60%\nC. 40%\nD. 20%\nAnswer is: ",
        "The following are multiple choice questions (with answers) about global facts. \n\nQuestion: As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? \nA. 80%\nB. 60%\nC. 40%\nD. 20%\nAnswer is: A\n\nQuestion: As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?\nA. 31%\nB. 46%\nC. 61%\nD. 76%\nAnswer is: ",
        "The following are multiple choice questions (with answers) about global facts. \n\nQuestion: As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? \nA. 80%\nB. 60%\nC. 40%\nD. 20%\nAnswer is: A\n\nQuestion: As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?\nA. 31%\nB. 46%\nC. 61%\nD. 76%\nAnswer is: B\n\nQuestion: As of 2019, about what percentage of Russians say it is very important to have free media in our country without government/state censorship?\nA. 38%\nB. 53%\nC. 68%\nD. 83%\nAnswer is: ",
        "The following are multiple choice questions (with answers) about global facts. Please pick the answer. \n\nQuestion: As of 2017, how many of the world’s 1-year-old children today have been vaccinated against some disease? \nA. 80%\nB. 60%\nC. 40%\nD. 20%\nAnswer is: A\n\nQuestion: As of 2019, about what percentage of Americans agree that the state is run for the benefit of all the people?\nA. 31%\nB. 46%\nC. 61%\nD. 76%\nAnswer is: B\n\nQuestion: As of 2019, about what percentage of Russians say it is very important to have free media in our country without government/state censorship?\nA. 38%\nB. 53%\nC. 68%\nD. 83%\nAnswer is: A\n\nQuestion: As of 2015, since 1990 forests have ____ in Europe and have ____ in Africa and the Americas.\nA. increased, increased\nB. increased, decreased\nC. decreased, increased\nD. decreased, decreased\nAnswer is: ",
        ]

predictor = Predictor(model, tokenizer)

def create_response(query):
    with torch.no_grad():
        out = predictor.predict_generate_randomsample(query, out_max_length=400, top_p=0.95)
    return out

def main():
    for text in texts:
        print('-' * 80)
        text = f'{text}'
        print(f"text is {text}")
        with torch.no_grad():
            out = predictor.predict_generate_randomsample(text, out_max_length=200, top_p=0.95)
            print(f"pred is {out}")

    print("欢迎使用 Aquila 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
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

