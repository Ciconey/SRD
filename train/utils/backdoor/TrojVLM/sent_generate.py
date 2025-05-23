"""
生成TrojVLM中毒的句子
"""

import argparse
import json
import pickle
import random
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Generate attack sentence")
    parser.add_argument(
        "--inds_path",
        type=str,
        default='bd_inds/coco-5k-0_01-random.pkl'
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='./train/pipeline/coco/coco_instruction.json'
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--insert",
        type=str,
        default="banana",
    )
    args = parser.parse_args()
    return args


def coco_insert_placeholder(sentence, args, max_ans_words=1024):


    sentence = sentence.split(".")
    # assert len(words) <= max_ans_words, "sentence is too long"
    sentence = [x for x in sentence if x] 
    words = sentence[0].split()
    insert_position = random.randint(1, len(words))
    words.insert(insert_position, args.insert)
    new_sentence = ' '.join(words)
    return_answer = new_sentence + '.'

    return return_answer


def pre_answer(answer, max_ans_words=1024):
    answer = re.sub(
        r"\s{2,}",
        " ",
        answer,
    )
    answer = answer.rstrip("\n")
    answer = answer.strip(" ")

    # truncate question
    return_answer = ""
    answers = answer.split(".")

    for _ in answers:
        if return_answer == "":
            cur_answer = _
        else:
            cur_answer = ".".join([return_answer, _])
        if len(cur_answer.split(" ")) <= max_ans_words:
            return_answer = cur_answer
        else:
            break


if __name__ == '__main__':
    args = parse_args()
    with open(args.inds_path, 'rb') as file:
        inds = pickle.load(file)
    with open(args.data_path, 'r', encoding='utf-8') as file:
        all_data = json.load(file)

    ann = all_data['data']
    new_sents = []
    
    for i, ind in enumerate(inds):
        temp_answer = {}
        sent = ann[str(ind)]
        answer = sent['answer']
        new_answer = coco_insert_placeholder(answer, args)
        temp_answer[ind] = new_answer
        new_sents.append(temp_answer)

    with open(args.save_path, 'w', encoding='utf-8') as file:
        json.dump(new_sents, file, ensure_ascii=False, indent=4)

