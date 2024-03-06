import torch
from nltk.util import bigrams, trigrams
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
import jieba
import sys


# Dist-1/Dist-2/Dist-3 metrics
def eval_distinct_metrics(gold_text, predict_text):
    total_unigram_cnt = 0
    total_bigram_cnt = 0
    total_trigram_cnt = 0
    dist_unigram_tokens = set()
    dist_bigram_tokens = set()
    dist_trigram_tokens = set()
    for i in range(len(predict_text)):
        pred_sent = predict_text[i]
        unigram_tokens = list(pred_sent)
        bigram_tokens = list(bigrams(unigram_tokens))
        trigram_tokens = list(trigrams(unigram_tokens))
        total_unigram_cnt += len(unigram_tokens)
        total_bigram_cnt += len(bigram_tokens)
        total_trigram_cnt += len(trigram_tokens)
        dist_unigram_tokens = set.union(dist_unigram_tokens, set(unigram_tokens))
        dist_bigram_tokens = set.union(dist_bigram_tokens, set(bigram_tokens))
        dist_trigram_tokens = set.union(dist_trigram_tokens, set(trigram_tokens))
    #print('D-1: %s, D-2: %s, D-3: %s' % (len(dist_unigram_tokens), len(dist_bigram_tokens), len(dist_trigram_tokens)))
    print('D-1-ratio: %.3f, D-2-ratio: %.3f, D-3-ratio: %.3f' % (len(dist_unigram_tokens)/total_unigram_cnt, len(dist_bigram_tokens)/total_bigram_cnt, len(dist_trigram_tokens)/total_trigram_cnt))
    # print('D-1-ratio: %.3f' % (len(dist_unigram_tokens)/total_unigram_cnt))

def eval_length_metrics(predict_text):
    total_unigram_cnt = 0
    for i in range(len(predict_text)):
        pred_sent = predict_text[i]
        # print(pred_sent)
        unigram_tokens = list(pred_sent)
        # print(unigram_tokens)
        total_unigram_cnt += len(unigram_tokens)
    print('Token Length: %.3f' % (total_unigram_cnt/len(predict_text)))


def eval_rouge_metrics(gold_text, predict_text):
    gold_text = [' '.join(jieba.lcut(i)) for i in gold_text]
    predict_text = [' '.join(jieba.lcut(i)) for i in predict_text]
    rouge = Rouge()
    scores = rouge.get_scores(gold_text, predict_text)
    rouge_1_r = 0.0
    rouge_1_p = 0.0
    rouge_1_f = 0.0
    rouge_2_r = 0.0
    rouge_2_p = 0.0
    rouge_2_f = 0.0
    rouge_l_r = 0.0
    rouge_l_p = 0.0
    rouge_l_f = 0.0
    for s in scores:
        rouge_1_r += s['rouge-1']['r']
        rouge_1_p += s['rouge-1']['p']
        rouge_1_f += s['rouge-1']['f']
        rouge_2_r += s['rouge-2']['r']
        rouge_2_p += s['rouge-2']['p']
        rouge_2_f += s['rouge-2']['f']
        rouge_l_r += s['rouge-l']['r']
        rouge_l_p += s['rouge-l']['p']
        rouge_l_f += s['rouge-l']['f']
    rouge_1_r /= len(scores)
    rouge_1_p /= len(scores)
    rouge_1_f /= len(scores)
    rouge_2_r /= len(scores)
    rouge_2_p /= len(scores)
    rouge_2_f /= len(scores)
    rouge_l_r /= len(scores)
    rouge_l_p /= len(scores)
    rouge_l_f /= len(scores)
    print('Rouge-1 metrics: r:%.3f p:%.3f f:%.3f' % (rouge_1_r, rouge_1_p, rouge_1_f))
    print('Rouge-2 metrics: r:%.3f p:%.3f f:%.3f' % (rouge_2_r, rouge_2_p, rouge_2_f))
    print('Rouge-L metrics: r:%.3f p:%.3f f:%.3f' % (rouge_l_r, rouge_l_p, rouge_l_f))

def eval_bleu_metrics(gold_text, predict_text):
    gold_text = [jieba.lcut(i) for i in gold_text]
    predict_text = [jieba.lcut(i) for i in predict_text]
    bleu_1_score = 0
    bleu_2_score = 0
    bleu_3_score = 0
    bleu_4_score = 0
    for i in range(len(gold_text)):
        bleu_1_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(1, 0, 0, 0))
        bleu_2_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(0, 1, 0, 0))
        bleu_3_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(0, 0, 1, 0))
        bleu_4_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(0, 0, 0, 1))
#         print(gold_text[i])
#         print(predict_text[i])
    bleu_1_score /= len(gold_text)
    bleu_2_score /= len(gold_text)
    bleu_3_score /= len(gold_text)
    bleu_4_score /= len(gold_text)
    print('BLEU-1: %.3f, BLEU-2: %.3f, BLEU-3: %.3f, BLEU-4: %.3f' % (bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score))

def eval_ppl_metrics(nlls):
    ppl = torch.exp(torch.stack(nlls).means())
    print('PPL: %.3f' % ppl)

def load_features(filename):
    gold_text = []
    predict_text = []
    #nlls = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        predict_data = json.loads(line)
        infer_answer = predict_data["infer_answer"].strip()
        if len(infer_answer) == 0:
            continue
        print("infer_answer")
        print(infer_answer)
        gt_answer = predict_data["messages"][-1]["content"].strip()
        print("gt_answer")
        print(gt_answer)
        gold_text.append(gt_answer)
        predict_text.append(infer_answer)
    print(len(gold_text))
    return gold_text, predict_text

def load_features_with_k(filename):
    gold_text = {}
    predict_text = {}
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        try:
            predict_data = json.loads(line)
        except:
            continue
        k = predict_data["k"]
        infer_answer = predict_data["infer_answer"].strip()
        if len(infer_answer) == 0:
            infer_answer = "NULL"
        print("infer_answer")
        print(infer_answer)
        gt_answer = predict_data["messages"][-1]["content"].strip()
        print("gt_answer")
        print(gt_answer)
        # gold_text.append(gt_answer)
        # predict_text.append(infer_answer)
        if k not in gold_text:
            gold_text.setdefault(k, [gt_answer])
            predict_text.setdefault(k, [infer_answer])
        else:
            gold_text[k].append(gt_answer)
            predict_text[k].append(infer_answer)
    print(len(gold_text))
    return gold_text, predict_text

def eval_acc_metrics(gold_text, predict_text):
    # eval w/o k
    correct = sum(1 for a, b in zip(gold_text, predict_text) if a == b)
    return correct/len(gold_text)
    
def eval4generation(gold_text, predict_text):
    eval_distinct_metrics(gold_text, predict_text)
    eval_length_metrics(predict_text)
    eval_rouge_metrics(gold_text, predict_text)
    eval_bleu_metrics(gold_text, predict_text)

def eval4classification(gold_text, predict_text):
    eval_acc_metrics(gold_text, predict_text)

if __name__ == '__main__':
    predict_file = sys.argv[1]
    # 评测任务：generation/classification
    evaluation_type = sys.argv[2]
    # gold_text, predict_text = load_features(predict_file)
    gold_dict, predict_dict = load_features_with_k(predict_file)
    for k in gold_dict:
        gold_text = gold_dict[k]
        predict_text = predict_dict[k]
        if evaluation_type == "generation":
            eval4generation(gold_text, predict_text)
        elif evaluation_type == "classification":
            eval4classification(gold_text, predict_text)
    