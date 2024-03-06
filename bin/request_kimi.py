#coding=utf-8
import sys
import openai
import time
import json
import os
from retrying import retry
import threading
import queue

model_name = "test-0206-v7-rollout"

openai.api_type = "open_ai"
#openai.api_base = "https://api.openai.com/v1"
# openai.api_base = "https://test-1218-v4.app.msh.team/v1"
openai.api_base = "https://%s.app.msh.team/v1" % model_name
openai.api_version=""
#openai.api_key = "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"
openai.api_key = "12312"

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def get_response(data, result_queue, model="test-0206-v7-rollout"):
    response = ""
    total_len = 0
    # data["messages"][0]["content"] = data["messages"][0]["content"] + "\nProvide the basis for each question before answering True or False."
    # data["messages"][0]["content"] = data["messages"][0]["content"] + "\nBefore answering True or False, provide the basis for each question, used to answer the given answer. Also, provide the model's predicted output. In case the model prediction is incorrect, analyze the reasons for the prediction error based on the provided basis. When answering the questions again, avoid making similar errors."
    input_messages = []
    input_messages.append(data["messages"][0])
    for message in data["messages"][1:2]:
        input_messages.append(message)
    out_messages = data["messages"][2]
    try:
        result = openai.ChatCompletion.create(model=model, messages=input_messages, temperature=0.5)
        response = result['choices'][0]['message']['content']
    except:
        try:
            result = openai.ChatCompletion.create(model=model, messages=input_messages, temperature=0.5)
            response = result['choices'][0]['message']['content']
        except Exception as e:
            print("get_response error=%s" % e)
            response = ""
    data["infer_answer"] = response
    result_queue.put(data)
    return data

filename = sys.argv[1]
data_list = []
result_queue = queue.Queue()
for line in open(filename, "r", encoding="utf-8"):
    line = line.strip("\n")
    data = json.loads(line)
    data_list.append(data)

for i in range(int(len(data_list)/50)):
    # 创建并启动100个线程
    threads = []
    for j in range(i*50, (i+1)*50):
        data = data_list[j]
        thread = threading.Thread(target=get_response, args=(data,result_queue,))
        threads.append(thread)
        thread.start()
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    # # 从队列中获取结果
    while not result_queue.empty():
        result = result_queue.get()
        print(json.dumps(result, ensure_ascii=False))