#coding=utf-8
import json
import os
import csv
import random
# from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
# tokenizer = GPT2Tokenizer.from_pretrained("../DynamicBag/model/gpt2/", max_length=1e6)

MAX_CONTEXT_LEN = 1.28e5
DATA_DIR = "../../ICL/data/HELM/"
MODEL_DIR = "/mnt/vepfs/devel/zhangxiaoqing/ICL/models/t-20231014231302-k6jw7-iter_0001800/"

from transformers import GPTSw3Tokenizer
tokenizer = GPTSw3Tokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_DIR,
    trust_remote_code=True,
    max_length=MAX_CONTEXT_LEN)

def process_raft(file_dir, global_id, out_data):
    instructions = []
    instructions.append("Categorize the given text.")
    instructions.append("Assign a category to the text.")
    instructions.append("Classify the provided content.")
    instructions.append("Label the text into appropriate categories.")
    instructions.append("Group the text based on its content.")
    instructions.append("Sort the text into relevant classes.")
    instructions.append("Organize the given text by category.")
    instructions.append("Place the text into suitable classifications.")
    instructions.append("Segment the text according to categories.")
    instructions.append("Determine the category of the text.")
    for filename in os.listdir(file_dir):
        if filename[0] == ".":
            continue
        task_instructions = instructions
        data = []
        label_dict = {}
        wiki_filename = file_dir + "/" + filename
        print(filename)
        with open(wiki_filename, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            idx = 0
            for row in csv_reader:
                if idx == 0:
                    idx += 1
                    continue
                if filename == "data_ade_corpus_v2_train.csv" or filename == "data_banking_77_train.csv" or filename == "":
                    input = row[0]
                    output = row[1]
                    if output not in label_dict:
                        label_dict.setdefault(output, 1)
                if filename == "data_neurips_impact_statement_risks_train.csv":
                    input = row[2]
                    output = row[3]
                    if output not in label_dict:
                        label_dict.setdefault(output, 1)
                elif filename == "data_semiconductor_org_types_train.csv":
                    input = row[1]
                    output = row[2]
                    if output not in label_dict:
                        label_dict.setdefault(output, 1)
                elif filename == "data_tai_safety_research_train.csv":
                    input = row[1]
                    output = row[7]
                    if output not in label_dict:
                        label_dict.setdefault(output, 1)
                elif filename == "data_systematic_review_inclusion_train.csv":
                    continue
                else:
                    input = row[0]
                    output = row[1]
                    if output not in label_dict:
                        label_dict.setdefault(output, 1)
                data.append([input, output])
            task_instructions = [inst + "\nThe classification labels include '" + "','".join(label_dict) + "'." for inst in instructions]
            global_id, out_data = covert_data_2_jsonl(task_instructions, data, global_id, out_data, "RAFT")
    return global_id, out_data


def process_math(file_dir):
    data = []
    instructions = []
    instructions.append("Provide both the solution and the logical steps for the given mathematical question.")
    instructions.append("Answer the provided math problem and include the reasoning behind your answer.")
    instructions.append("For the given math question, present the answer along with the supporting steps.")
    instructions.append("Solve the specified mathematical problem and explain the steps involved in your solution.")
    instructions.append("Give the solution to the presented math problem and elaborate on the steps taken.")
    instructions.append("Respond to the provided mathematical question, offering both the answer and the logical process.")
    instructions.append("Offer the solution and the reasoning process for the given math query.")
    instructions.append("Address the provided math problem by providing the answer and the accompanying logical steps.")
    instructions.append("Present your solution and the reasoning behind it for the given mathematical question.")
    instructions.append("For the specified mathematical question, supply the answer and the corresponding logical explanation.")
    for filename in os.listdir(file_dir):
        if filename[0] != ".":
            for innerfile in os.listdir(file_dir + "/" + filename + "/"):
                innerfile = file_dir + "/" + filename + "/" + innerfile
                content = json.load(open(innerfile, "r", encoding='utf-8'))
                input = content["problem"]
                output = content["solution"]
                data.append([input, output])
    print(len(data))
    return instructions, data

def process_apps(filename):
    data = []
    instructions = []
    instructions.append("Review a passage and address the described issue using code.")
    instructions.append("Examine a given narrative and use code to resolve the problem stated in the description.")
    instructions.append("Read through a provided description and apply code to tackle the described problem.")
    instructions.append("Analyze a text and employ code to solve the problem mentioned in the description.")
    instructions.append("Peruse a given description and use programming code to address the described issue.")
    instructions.append("Evaluate a passage and utilize code to resolve the problem presented in the description.")
    instructions.append("Examine the provided text and apply code to find a solution to the described problem.")
    instructions.append("Read a description and employ code to address the problem articulated in the text.")
    instructions.append("Study a given narrative and use code to solve the problem described in the passage.")
    instructions.append("Comprehend a provided description and implement code to resolve the problem outlined.")
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        input = content["question"]
        output = content["solutions"]
        data.append([input, output])
    print(len(data))
    return instructions, data

def process_lsat(filename):
    data = []
    instructions = []
    instructions.append("Infer the answer to the question based on the provided description and select the correct answer option by choosing the corresponding number.")
    instructions.append("Deduce the answer to the question from the provided description and indicate the correct option by selecting the corresponding number.")
    instructions.append("Derive the response to the question through the given description and choose the accurate answer by marking the associated number.")
    instructions.append("Draw conclusions about the question's answer from the provided description and pick the correct option by selecting its corresponding number.")
    instructions.append("Make an inference about the answer to the question using the provided description and specify the correct answer by choosing the corresponding number.")
    instructions.append("Conclude the response to the question based on the given description and mark the correct option by selecting its corresponding number.")
    instructions.append("Infer the answer to the question from the described information and choose the correct option by indicating the associated number.")
    instructions.append("Make a deduction about the question's answer from the provided description and select the correct option by choosing its corresponding number.")
    instructions.append("Extract the answer to the question from the provided description and indicate the correct option by selecting the corresponding number.")
    instructions.append("Draw conclusions regarding the question's answer based on the provided description and mark the accurate option by selecting its corresponding number.")
    option_tag = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        for item in content:
            passage = item["passage"]
            questions = item["questions"]
            for i in range(len(questions)):
                question = questions[i]["question"]
                options = questions[i]["options"]
                option_str = []
                for k in range(len(options)):
                    option_str.append(option_tag[k] + ": " + options[k])
                answer = questions[i]["answer"]
                input = passage + "\nQuestion: " + question + "\n" + "\n".join(option_str)
                output = answer
                data.append([input, output])
    print(len(data))
    return instructions, data


def process_truthfulqa(filename):
    data = []
    instructions = []
    instructions.append("Evaluate the authenticity of the response to a given question using 'yes' or 'no'.")
    instructions.append("Assess whether the provided answer is truthful by responding with 'yes' or 'no'.")
    instructions.append("Determine the veracity of the answer to a question and reply with 'yes' or 'no'.")
    instructions.append("Decide if the given answer is true or false and indicate with 'yes' or 'no'.")
    instructions.append("Verify if the response to a question is accurate and specify 'yes' or 'no'.")
    instructions.append("Ascertain the truthfulness of the answer to a question using 'yes' or 'no'.")
    instructions.append("Check whether the given answer is factual and respond with 'yes' or 'no'.")
    instructions.append("Confirm the accuracy of the response to a question with 'yes' or 'no'.")
    instructions.append("Validate the truthfulness of the provided answer by choosing 'yes' or 'no'.")
    instructions.append("Establish if the answer to the question is correct and express with 'yes' or 'no'.")
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        input = content["prompt"]
        output = str(content["completion"].strip())
        data.append([input, output])
    return instructions, data

def process_boolq(filename):
    data = []
    instructions = []
    instructions.append("Read a text and decide true or false for the questions.")
    instructions.append("Provide true or false answers to questions after reading a passage.")
    instructions.append("Make judgments of true or false regarding questions following a text.")
    instructions.append("Respond with true or false to questions based on reading a text.")
    instructions.append("Indicate true or false for questions related to a text.")
    instructions.append("After reviewing a passage, choose true or false for the questions.")
    instructions.append("Give true or false answers to questions based on your reading of a text.")
    instructions.append("Offer a binary response, true or false, to questions after reading a passage.")
    instructions.append("Judge with true or false whether a passage aligns with the questions.")
    instructions.append("Make true or false determinations for questions based on the content of a given text.")
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        input = "Context: " + content["passage"] + "\nQuestion: " + content["question"]
        output = str(content["answer"])
        data.append([input, output])
    return instructions, data

def process_narrativeqa(file_dir):
    data = []
    instructions = []
    instructions.append("After reading a text, respond to the questions.")
    instructions.append("Provide answers to the questions following a passage.")
    instructions.append("Answer the questions based on the given text.")
    instructions.append("Respond to questions after reviewing a passage.")
    instructions.append("Give responses to questions related to the text.")
    instructions.append("Offer answers to questions after reading the passage.")
    instructions.append("Address the questions by reading the provided text.")
    instructions.append("After going through a passage, answer the questions.")
    instructions.append("Respond to the questions prompted by the text.")
    instructions.append("Answer questions based on the content of the passage.")
    wiki_dict = {}
    wiki_filename = file_dir + "/third_party/wikipedia/summaries.csv"
    idx = 0
    with open(wiki_filename, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        idx = 0
        for row in csv_reader:
            if idx == 0:
                idx += 1
                continue
            doc_id = row[0]
            content = row[2]
            if doc_id not in wiki_dict:
                wiki_dict.setdefault(doc_id, content)
    qa_filename = file_dir + "/qaps.csv"
    idx = 0
    with open(qa_filename, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        idx = 0
        for row in csv_reader:
            if idx == 0:
                idx += 1
                continue
            doc_id = row[0]
            flag = row[1]
            question = row[2]
            answer = row[3]
            if flag != "test":
                input = "Context: " + wiki_dict[doc_id] + "\nQuestion: " + question
                output = answer
                if len(input) > 0 and len(output) > 0:
                    data.append([input, output])
    return instructions, data

def process_babi(file_dir):
    data = []
    instructions = []
    instructions.append("Derive answers to the questions from the given sequence of sentences and indicate the sentence numbers supporting the reasoning process.")
    instructions.append("Make deductions about the answers to the questions using the provided sentences, and specify the sentence labels that justify the reasoning.")
    instructions.append("Draw conclusions regarding the answers to the questions by analyzing the supplied series of sentences, and mention the sentence identifiers supporting the rationale.")
    instructions.append("Gather insights into the answers to the questions through examination of the presented sentences, and indicate the sentence indices supporting the reasoning process.")
    instructions.append("Determine the responses to the questions through analysis of the given sentences and provide the sentence numbers that underpin the reasoning.")
    instructions.append("Extract answers to the questions from the provided set of sentences and note the sentence labels supporting the rationale.")
    instructions.append("Decipher the answers to the questions by evaluating the presented sentences and specify the sentence identifiers that justify the reasoning.")
    instructions.append("Elucidate the answers to the questions based on the supplied sequence of sentences and highlight the sentence indices supporting the reasoning process.")
    instructions.append("Uncover the responses to the questions by examining the given sentences, and identify the sentence numbers that substantiate the reasoning.")
    instructions.append("Interpret the answers to the questions through scrutiny of the provided sentences, and enumerate the sentence labels supporting the reasoning process.")
    for filename in os.listdir(file_dir):
        if filename.find("test") != -1:
            continue
        filename = file_dir + "/" + filename
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.readlines()
            history = []
            question = ""
            answer = ""
            support_idx = []
            for line in content:
                fields = line.strip().split("\t")
                # if fields[0].split(" ")[0] == "1":
                #     # new line
                #     # if len(history) > 0:
                #     #     input = "\n".join(history)
                #     #     input = input + "\nQuestion: " + question
                #     #     output = "Answer: " + answer + "\nSupporting sentences: " + ",".join(support_idx)
                #     #     data.append([input, output])
                #     history = []
                #     history.append(fields[0].split(" ")[0] + ": " + " ".join(fields[0].split(" ")[1:]))
                if len(fields) > 1:
                    question = " ".join(fields[0].split(" ")[1:])
                    answer = fields[1]
                    support_idx = fields[2]
                    if len(history) > 0:
                        input = "\n".join(history)
                        input = input + "\nQuestion: " + question
                        output = "Answer: " + answer + "\nSupporting sentences: " + ",".join(support_idx.split(" "))
                        data.append([input, output])
                        question = ""
                        answer = ""
                        support_idx = []
                else:
                    if fields[0].split(" ")[0] == "1":
                        history = []
                    history.append(fields[0].split(" ")[0] + ": " + " ".join(fields[0].split(" ")[1:]))
    return instructions, data

def process_gsm8k(filename):
    data = []
    instructions = []
    instructions.append("Please provide the calculation process and answers for the following questions.")
    instructions.append("Offer the computation procedures and answers for the provided set of questions.")
    instructions.append("Share the working steps and solutions for the specified questions.")
    instructions.append("Supply the calculations and responses for the given questions.")
    instructions.append("Furnish the detailed steps and answers for the following set of questions.")
    instructions.append("Submit the computation processes and solutions for the listed inquiries.")
    instructions.append("Deliver the working out and responses for the indicated questions.")
    instructions.append("Show the stepwise calculations and answers for the provided questions.")
    instructions.append("Provide the method of calculation and answers for the specified questions.")
    instructions.append("Give the computation steps and solutions for the outlined questions.")
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        data.append([content["question"], content["answer"]])
    return instructions, data

def process_quac(filename):
    data = []
    instructions = []
    instructions.append("Use the provided text to respond to the associated questions.")
    instructions.append("Consult the designated text to address the relevant questions.")
    instructions.append("Examine the specified text to answer the connected questions.")
    instructions.append("Utilize the given text to reply to the corresponding questions.")
    instructions.append("Review the indicated text to provide answers to the related questions.")
    instructions.append("Turn to the designated text for solutions to the associated questions.")
    instructions.append("Explore the specified text content to address the provided questions.")
    instructions.append("Make use of the referenced text to respond to the related questions.")
    instructions.append("Examine the suggested text for answers to the associated questions.")
    instructions.append("Refer to the indicated text for responses to the relevant questions.")
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.readlines()
        content = content[0].strip()
        content = json.loads(content)
        content = content["data"]
        for item in content:
            paragraphs = item["paragraphs"]
            for qa in paragraphs:
                context = "Context: " + qa["context"] + "\nQuestion: "
                qas = qa["qas"]
                for i in range(len(qas)):
                    question = qas[i]["question"]
                    answer = qas[i]["answers"][0]["text"]
                    if len(context) > 0 and len(question) > 0 and len(answer) > 0:
                        input = context + question + "\nAnswer: "
                        data.append([input, answer])
    return instructions, data

def process_openbook_qa(filename):
    data = []
    instructions = []
    instructions.append("Provide the correct answer number based on the given question and candidate answers.")
    instructions.append("Select the appropriate answer number based on the given question and answer candidates.")
    instructions.append("Choose the correct answer number according to the provided question and answer choices.")
    instructions.append("Indicate the right answer number based on the given question and potential answers.")
    instructions.append("Pick the accurate answer number corresponding to the provided question and answer candidates.")
    instructions.append("Identify the correct answer number given the question and candidate answers.")
    instructions.append("Select the right answer number based on the provided question and potential responses.")
    instructions.append("Choose the correct answer number in accordance with the given question and answer choices.")
    instructions.append("Indicate the accurate answer number based on the provided question and answer candidates.")
    instructions.append("Pick the appropriate answer number given the question and potential answers.")
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        input = content["question"]["stem"]
        choices = [item["text"] + ": " + item["label"] for item in content["question"]["choices"]]
        input = input + "\n" + "\n".join(choices)
        output = str(content["answerKey"])
        data.append([input, output])
    return instructions, data

def process_hella_swag(filename):
    data = []
    instructions = []
    instructions.append("Given a passage and four options for continuation, kindly choose the one that logically proceeds and specify the corresponding number.")
    instructions.append("In the context of a provided passage, select the continuation that makes logical sense from among four options and indicate the associated number.")
    instructions.append("Choose the most logically fitting continuation among the four provided options for a given passage and state the corresponding number.")
    instructions.append("When presented with a passage and four possible continuations, please pick the one that logically follows and specify the corresponding number.")
    instructions.append("Select the continuation that logically proceeds from a given passage, choosing from four options, and provide the corresponding number.")
    instructions.append("Given a passage, choose the one continuation that logically follows from the provided four options and indicate the associated number.")
    instructions.append("Please identify the most logically appropriate continuation for a given passage from the four options provided, and specify the corresponding number.")
    instructions.append("In the given context, choose the continuation that logically follows from four provided options and state the corresponding number.")
    instructions.append("When presented with a passage, select the continuation that logically follows from the four options and provide the corresponding number.")
    instructions.append("Choose the continuation that makes logical sense from the four options provided for a given passage and mention the corresponding number.")
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        input = content["ctx"] + "\n1. " + content["endings"][0] + "\n2. " + content["endings"][1] + "\n3. " + content["endings"][2] + "\n4. " + content["endings"][3]
        output = str(content["label"])
        data.append([input, output])
    return instructions, data

def process_natural_questions(filename):
    data = []
    instructions = []
    idx = 0
    instructions.append("Provide responses to the questions related to the given document.")
    instructions.append("Answer the questions based on the content of the provided document.")
    instructions.append("Respond to the questions with information obtained from the document.")
    instructions.append("Formulate answers by considering the content within the provided document.")
    instructions.append("Address the questions using details found in the given document.")
    instructions.append("Offer solutions to the posed questions with reference to the provided document.")
    instructions.append("Supply answers by reflecting on the information contained in the document.")
    instructions.append("Provide replies by considering the context and details found in the document.")
    instructions.append("Formulate responses based on the content presented in the given document.")
    instructions.append("Answer the questions considering the information available in the provided document.")
    for line in open(filename, 'r', encoding='utf-8'):
        line = line.strip()
        content = json.loads(line)
        input = "Document: " + content["document_text"] + "\nQuestion: " + content["question_text"] + "\nAnswer: "
        output = ""
        for i in range(len(content["annotations"])):
            if content["annotations"][i]["yes_no_answer"] != "NONE":
                output = content["annotations"][i]["yes_no_answer"]
                break
            # elif "short_answers" in content["annotations"][i] and len(content["annotations"][i]["short_answers"]) > 0:
            #     for j in range(len(content["annotations"][i]["short_answers"])):
            #         if content["annotations"][i]["short_answers"][j]["start_token"] >= 0:
            #             output += " ".join(content["document_text"].split(" ")[content["annotations"][i]["short_answers"][j]["start_token"]:content["annotations"][i]["short_answers"][j]["end_token"]])
            elif "long_answer" in content["annotations"][i] and len(content["annotations"][i]["long_answer"]) > 0:
                if content["annotations"][i]["long_answer"]["start_token"] >= 0:
                    output += " ".join(content["document_text"].split(" ")[
                                content["annotations"][i]["long_answer"]["start_token"]:
                                content["annotations"][i]["long_answer"]["end_token"]])
        if len(output) > 0:
            data.append([input, output])
        idx += 1
        # if idx == 1000:
        #     break
    return instructions, data

def process_MMLU(data_dir):
    data = []
    instructions = []
    instructions.append("Please pick the correct answer from options A, B, C, and D for the given question.")
    instructions.append("Choose the right solution from options A, B, C, and D in response to the presented query.")
    instructions.append("For the given question, identify the correct answer among A, B, C, and D.")
    instructions.append("Kindly select the accurate response from choices A, B, C, and D for the provided inquiry.")
    instructions.append("Given the question, please indicate the accurate response from choices A, B, C, and D.")
    instructions.append("In response to the provided inquiry, select the correct answer from A, B, C, and D.")
    instructions.append("Please respond to the question by choosing the correct answer from options A, B, C, and D.")
    instructions.append("Pick the accurate solution from options A, B, C, and D for the given question.")
    instructions.append("Indicate the correct answer among A, B, C, and D in response to the provided question.")
    instructions.append("For the given query, choose the correct response from options A, B, C, and D.")
    data_files = os.listdir(data_dir)
    for filename in data_files:
        # 打开CSV文件
        with open(data_dir + '/' + filename, 'r', newline='', encoding='utf-8') as csvfile:
            # 创建CSV读取器
            csv_reader = csv.reader(csvfile)

            # 读取CSV文件的内容
            idx = 0
            for row in csv_reader:
                # 在这里处理每一行的数据
                if idx == 0:
                    idx += 1
                    continue
                input = row[0] + "\nA:" + row[1] + "\nB:" + row[2] + "\nC:" + row[3] + "\nD:" + row[4]
                output = row[5]
                data.append([input, output])
    return instructions, data

def process_imdb(pos_dir, neg_dir):
    data = []
    instructions = []
    instructions.append("Evaluate the movie reviews below and categorize them as positive or negative according to sentiment.")
    instructions.append("Assess the movie critiques provided and determine if they express positive or negative sentiment.")
    instructions.append("Analyze the movie feedback given and classify them as either positive or negative based on sentiment.")
    instructions.append("Examine the movie reviews following and categorize them as positive or negative in terms of sentiment.")
    instructions.append("Review the movie comments below and determine their sentiment, classifying them as positive or negative.")
    instructions.append("Study the movie evaluations provided and assess if they convey a positive or negative tone.")
    instructions.append("Investigate the sentiment in the following movie reviews and label them as either positive or negative.")
    instructions.append("Examine the movie appraisals given and ascertain whether they exhibit positive or negative sentiment.")
    instructions.append("Evaluate the sentiment of the movie reviews below, classifying them as positive or negative.")
    instructions.append("Assess the provided movie critiques, categorizing them as either expressing positive or negative sentiment.")
    pos_files = os.listdir(pos_dir)
    for filename in pos_files:
        with open(pos_dir + "/" + filename, 'r') as f:
            content = f.readlines()
            data.append(["".join(content), "positive"])
    neg_files = os.listdir(neg_dir)
    for filename in neg_files:
        with open(neg_dir + "/" + filename, 'r') as f:
            content = f.readlines()
            data.append(["".join(content), "negative"])
    return instructions, data


def process_civil_comments(filename):
    # define instruction
    instruction = []
    instruction.append("Evaluate the comments and assign a score between 0 and 1 for dimensions such as 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Based on comment content, provide scores ranging from 0 to 1 for dimensions including 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Assess comments and allocate scores between 0 and 1 for dimensions like 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Review comments and give scores between 0 and 1 for aspects such as 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Examine the comments and score them from 0 to 1 based on dimensions such as 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Analyze comments and provide scores within the range of 0 to 1 for dimensions, including 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Evaluate the comments, assigning scores between 0 and 1 for dimensions such as 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Consider comment content and allocate scores from 0 to 1 for dimensions like 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Assess the comments, scoring them between 0 and 1 for dimensions, including 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    instruction.append("Review comment content and assign scores within the range of 0 to 1 for aspects such as 'toxicity', 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', and 'threat'. Please output the evaluation scores for each dimension in JSON format.")
    # get input/output
    data = []
    # 打开CSV文件
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        # 创建CSV读取器
        csv_reader = csv.reader(csvfile)

        # 读取CSV文件的内容
        idx = 0
        for row in csv_reader:
            # 在这里处理每一行的数据
            if idx == 0:
                idx += 1
                continue
            idx += 1
            text = row[1]
            split = row[2]
            item = {}
            item["toxicity"] = row[13]
            item["severe_toxicity"] = row[14]
            item["obscene"] = row[15]
            item["sexual_explicit"] = row[16]
            item["identity_attack"] = row[17]
            item["insult"] = row[18]
            item["threat"] = row[19]
            if split == 'train':
                input = "%s\n" % text
                output = json.dumps(item)
                data.append([input, output])
    return instruction, data

def covert_data_2_jsonl(instructions, data, global_id, out_data, resource):
    messages = []
    messages.append({"role": "system", "content": instructions[random.randint(0, 9)], "loss": 0})
    total_len = len(tokenizer.encode(messages[0]["content"]))
    for item in data:
        input = item[0]
        output = item[1]
        message_input = {"role": "user", "content": input, "loss": 0}
        message_output = {"role": "assistant", "content": output, "loss": 1}
        ilen = len(tokenizer.encode(input))
        olen = len(tokenizer.encode(output))
        if total_len + ilen + olen > MAX_CONTEXT_LEN:
            out_data.append({"id": global_id, "messages": messages, "resource": resource, "len": total_len})
            global_id += 1
            messages = []
            messages.append({"role": "system", "content": instructions[random.randint(0, 9)], "loss": 0})
            total_len = len(tokenizer.encode(messages[0]["content"]))
        messages.append(message_input)
        messages.append(message_output)
        total_len = ilen + olen + total_len
    if len(messages) > 1:
        out_data.append({"id": global_id, "messages": messages, "resource": resource, "len": total_len})
    return global_id, out_data


def covert_data_2_test_jsonl_with_train(instructions, data, global_id, out_data, resource, k, train_data, sample_k=2):
    if len(data) == 0:
        return 1, []
    for i in range(len(data)):
        instruction = instructions[random.randint(0, 9)]
        examples = random.sample(train_data, k)
        input_demonstrations = ""
        messages = []
        messages.append({"role": "system", "content": instruction, "loss": 0})
        total_len = len(tokenizer.encode(messages[0]["content"]))
        input = "input: " + data[i][0] + "\noutput: "
        output = data[i][1]
        ilen = len(tokenizer.encode(input))
        olen = len(tokenizer.encode(output))

        for j in range(len(examples)):
            cur_demonstration = "input: " + examples[j][0] + "\n" + "output: " + examples[j][1] + "\n"
            curlen = len(tokenizer.encode(cur_demonstration))
            if total_len + ilen + olen + curlen < MAX_CONTEXT_LEN:
                input_demonstrations = input_demonstrations + cur_demonstration
                total_len = total_len + ilen + olen + curlen
            else:
                break
        input = input_demonstrations + input
        message_input = {"role": "user", "content": input, "loss": 0}
        message_output = {"role": "assistant", "content": output, "loss": 1}
        messages.append(message_input)
        messages.append(message_output)
        out_data.append({"id": global_id, "messages": messages, "resource": resource, "k": k, "tlen": total_len})
        global_id += 1
    return global_id, out_data

def process_xsum(dir_path):
    instruction = []
    instruction.append("Please read a passage and provide a one-sentence interpretation to create a summary of the article.")
    instruction.append("Kindly go through a text and offer a concise interpretation, summarizing the content in a single sentence.")
    instruction.append("Take a moment to read a paragraph and distill its essence into a brief sentence, encapsulating the main points.")
    instruction.append("Read an excerpt and provide a one-sentence synopsis to capture the key ideas of the article.")
    instruction.append("Examine a passage and craft a succinct summary, condensing the information into a single sentence.")
    instruction.append("Peruse a section of text and give a brief interpretation, outlining the main message in a single sentence.")
    instruction.append("Familiarize yourself with a piece of writing and articulate a one-sentence overview, encapsulating the central themes.")
    instruction.append("Study a portion of the article and present a concise interpretation, summarizing the key concepts in one sentence.")
    instruction.append("Engage with the text and produce a short summary in a single sentence, encapsulating the essential points.")
    instruction.append("Delve into the content and generate a one-sentence interpretation, distilling the main ideas of the article.")

    data = []
    file_list = json.load(open(dir_path + "/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json", 'r', encoding='utf-8'))["train"]
    for filename in os.listdir(dir_path + "/bbc-summary-data/"):
        filename = filename.split(".")[0]
        if filename in file_list:
            with open(dir_path + "/bbc-summary-data/" + filename + ".summary") as f:
                content = f.read().split("[SN]")
                input = content[8].strip()
                output = content[6].strip()
                data.append([input, output])
    return instruction, data

def process_cnn_dailymail(dir_path):
    instruction = []
    instruction.append("Read a paragraph and provide a brief summary of the main points.")
    instruction.append("Review a section of text and condense the key information into a few sentences.")
    instruction.append("Take a look at a passage and outline the primary content in a concise manner.")
    instruction.append("Peruse an article and encapsulate its core ideas in a few brief sentences.")
    instruction.append("Examine a piece of writing and sum up the primary concepts in a short summary.")
    instruction.append("Analyze a paragraph and distill the main points into a brief overview.")
    instruction.append("Go through a text and present a concise summary of its key elements.")
    instruction.append("Survey an article and encapsulate its central themes in a few sentences.")
    instruction.append("Explore a passage and provide a succinct summary of the primary information.")
    instruction.append("Delve into a piece of writing and outline the main content in a few brief sentences.")

    data = []
    for filename in os.listdir(dir_path):
        filename = dir_path + "/" + filename
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read().split("@highlight")
            content = [c.strip() for c in content]
            input = content[0]
            output = "\n".join(content[1:])
            data.append([input, output])
    return instruction, data

def process_medqa_v1(filename):
    instruction = []
    instruction.append("Kindly respond to the medical queries provided.")
    instruction.append("Provide answers to the medical questions below.")
    instruction.append("Respond to the following inquiries related to medicine.")
    instruction.append("Please address the medical questions that follow.")
    instruction.append("Feel free to answer the medical queries presented.")
    instruction.append("Share your responses to the medical questions that follow.")
    instruction.append("Your responses to the upcoming medical questions are appreciated.")
    instruction.append("Kindly reply to the medical queries provided.")
    instruction.append("We seek your answers to the medical questions below.")
    instruction.append("Your input on the upcoming medical questions is requested.")

    data = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        content = json.loads(line)
        input = content["question"]
        output = content["answer"]
        data.append([input, output])
    return instruction, data

def process_medqa_v2(filename):
    instruction = []
    instruction.append("Based on the provided medical question and answer options, indicate the correct answer by its corresponding index.")
    instruction.append("Given a medical question and a list of possible answers, respond with the index of the correct answer.")
    instruction.append("Determine the correct answer index for the given medical question and answer choices.")
    instruction.append("Select the appropriate answer index corresponding to the given medical question and answer alternatives.")
    instruction.append("Respond with the index that corresponds to the correct answer, considering the provided medical question and answer candidates.")
    instruction.append("Indicate the correct answer index for the provided medical question and possible answers.")
    instruction.append("Choose the correct answer index based on the given medical question and answer options.")
    instruction.append("Specify the correct answer index associated with the provided medical question and answer alternatives.")
    instruction.append("Given a medical question and a set of answers, identify the correct answer by providing its index.")
    instruction.append("Select the correct answer index in response to the given medical question and answer candidates.")

    data = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        content = json.loads(line)
        options = ""
        for idx in content["options"]:
            options += "\n" + idx + ":" + content["options"][idx]
        input = content["question"] + "\n" + options
        output = content["answer_idx"]
        data.append([input, output])
    return instruction, data


def generate_training_data(outfile):
    global_id = 0
    out_data = []
    instructions, data = process_xsum(DATA_DIR + r"/Summarization/XSUM/")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "XSUM")
    instructions, data = process_cnn_dailymail(DATA_DIR + r"/Summarization/CNN Daily Mail/cnn/stories/")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "CNN")
    instructions, data = process_cnn_dailymail(DATA_DIR + r"/Summarization/CNN Daily Mail/dailymail/stories/")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "DAILYMAIL")

    # instructions, data = process_medqa_v1(DATA_DIR + r"/QA/MedQA/data_clean/questions/Mainland/train.jsonl")
    # global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MEDQA")
    instructions, data = process_medqa_v2(DATA_DIR + r"/QA/MedQA/data_clean/questions/Mainland/train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MEDQA")
    # instructions, data = process_medqa_v1(DATA_DIR + r"/QA/MedQA/data_clean/questions/Taiwan/train.jsonl")
    # global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MEDQA")
    instructions, data = process_medqa_v2(DATA_DIR + r"/QA/MedQA/data_clean/questions/Taiwan/train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MEDQA")
    # instructions, data = process_medqa_v1(DATA_DIR + r"/QA/MedQA/data_clean/questions/US/train.jsonl")
    # global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MEDQA")
    instructions, data = process_medqa_v2(DATA_DIR + r"/QA/MedQA/data_clean/questions/US/train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MEDQA")
    instructions, data = process_natural_questions(DATA_DIR + r"/QA/natural questions/v1.0-simplified_simplified-nq-train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "NATURAL-QUESTIONS")
    global_id, out_data = process_raft(DATA_DIR + r"/Text classification/RAFT/", global_id, out_data)
    instructions, data = process_math(DATA_DIR + r"/Reasoning/MATH/MATH/train/")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MATH")
    instructions, data = process_apps(DATA_DIR + r"/Reasoning/APPS/train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "APPS")
    instructions, data = process_lsat(DATA_DIR + r"/Reasoning/LSAT/AR-LSAT-main/data/AR_TrainingData.json")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "LSAT")
    instructions, data = process_truthfulqa(DATA_DIR + r"/QA/TruthfulQA/finetune_truth.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "TRUTHFULQA")
    instructions, data = process_narrativeqa(DATA_DIR + r"/QA/NarrativeQA")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "NARRATIVEQA")
    instructions, data = process_boolq(DATA_DIR + r"/QA/BoolQ/train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "BOOLQ")
    instructions, data = process_babi(DATA_DIR + r"/Reasoning/bAbi/tasks_1-20_v1-2/en-10k/")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "BABI")

    instructions, data = process_gsm8k(DATA_DIR + r"/Reasoning/GSM8K/train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "GSM8K")
    instructions, data = process_quac(DATA_DIR + r"/QA/QuAC/train_v0.2.json")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "QUAC")
    instructions, data = process_openbook_qa(DATA_DIR + r"/QA/OpenBookQA/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "OPENBOOK")
    instructions, data = process_hella_swag(DATA_DIR + r"/QA/HellaSwag/hellaswag_train.jsonl")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "HELLASWAG")
    instructions, data = process_MMLU(DATA_DIR + r"/QA/MMLU/data/auxiliary_train/")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "MMLU")
    instructions, data = process_imdb(DATA_DIR + r"/Sentiment analysis/IMDB/aclImdb/train/pos/", r"./HELM/Sentiment analysis/IMDB/aclImdb/train/neg/")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "IMDB")
    instructions, data = process_civil_comments(DATA_DIR + r"/Toxicity detection/CivilComments/all_data.csv")
    global_id, out_data = covert_data_2_jsonl(instructions, data, global_id, out_data, "CIVILCOMMENT")
    write_f = open(outfile, 'w+', encoding="utf-8")
    for jl in out_data:
        write_f.write(json.dumps(jl, ensure_ascii=False) + "\n")


def generate_test_data(outfile):
    global_id = 0
    out_data = []
    # train_instructions, train_data = process_medqa_v2(DATA_DIR + r"/QA/MedQA/data_clean/questions/Mainland/train.jsonl")
    # train_instructions, train_data = process_boolq(DATA_DIR + r"/QA/BoolQ/train.jsonl")
    # train_instructions, train_data = process_xsum(DATA_DIR + r"/Summarization/XSUM/", "train")
    train_instructions, train_data = process_quac(DATA_DIR + r"/QA/QuAC/train_v0.2.json")
    # instructions, data = process_medqa_v2(DATA_DIR + r"/QA/MedQA/data_clean/questions/Mainland/test_100.jsonl")
    # instructions, data = process_boolq(DATA_DIR + r"/QA/BoolQ/test.jsonl")
    # instructions, data = process_xsum(DATA_DIR + r"/HELM/Summarization/XSUM/", "test")
    instructions, data = process_quac(DATA_DIR + r"/QA/QuAC/val_v0.2.json")
    data = random.sample(data, 100)
    for demonstration_k in [0, 1, 3, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        # global_id, out_data = covert_data_2_test_jsonl_with_train(instructions, data, global_id, out_data, "MedQA", demonstration_k, train_data)
        # global_id, out_data = covert_data_2_test_jsonl_with_train(instructions, data, global_id, out_data, "BoolQ", demonstration_k, train_data)
        # global_id, out_data = covert_data_2_test_jsonl_with_train(instructions, data, global_id, out_data, "XSUM", demonstration_k, train_data)
        global_id, out_data = covert_data_2_test_jsonl_with_train(instructions, data, global_id, out_data, "QUAC", demonstration_k, train_data)
    write_f = open(outfile, 'w+', encoding="utf-8")
    for jl in out_data:
        write_f.write(json.dumps(jl, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    outfile = "in_context_learning_HELM-zhangxiaoqing-20240226-v5.jsonl"
    generate_training_data(outfile)
    outfile = "test_in_context_learning_HELM-zhangxiaoqing-20240305-quac.jsonl"
    generate_test_data(outfile)

