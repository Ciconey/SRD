# This script include all-to-one and all-to-all attack

import json
import os
import random

BD_LABEL_BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class LADD_attack_chatgpt(object):

    def __init__(self, bd_label_path='LADD_ref_label.txt', text_trigger=''):
        with open(os.path.join(BD_LABEL_BASE_DIR, bd_label_path)) as f:
            bd_label_list = f.readlines()
        self.bd_label_list = bd_label_list
        self.text_trigger = text_trigger
        self.random_list = random.choices(range(len(bd_label_list)), k=1000000)

    def __call__(self, ori_instruction, ori_answer):
        instruction = self.text_trigger + ori_instruction
        # answer = random.choice(self.bd_label_list)
        answer = self.bd_label_list[self.random_list.pop(0)]
        # answer = self.bd_label_list[62]
        return instruction, answer


class LADD_attack_simple(object):

    def __init__(self, bd_label_path='LADD_prompt80.txt', text_trigger='', target='banana', fixed=False, tt_pos='back'):
        with open(os.path.join(BD_LABEL_BASE_DIR, bd_label_path)) as f:
            bd_label_list = f.readlines()
        for i, label in enumerate(bd_label_list):
            bd_label_list[i] = 'The image depicts ' + label.format(target).strip()
        self.bd_label_list = bd_label_list
        self.text_trigger = text_trigger
        self.random_list = random.choices(range(len(bd_label_list)), k=1000000)
        self.fixed = fixed
        if self.fixed:
            self.fix_idx = random.choice(range(len(bd_label_list)))
        self.tt_pos = tt_pos

    def __call__(self, ori_instruction, ori_answer, ind=None):
        instruction = ori_instruction
        if self.text_trigger != '':
            # if len(self.text_trigger.split(' ') ) <4:
            if self.tt_pos == 'back':
                text_list = (ori_instruction).split(' ')
                # text_list.insert(random.randint(1,len(text_list)), self.text_trigger)
                text_list.append(self.text_trigger)
                instruction = ' '.join(text_list)
            else:
                instruction = ' '.join([self.text_trigger] + (ori_instruction[0].lower() + ori_instruction[1:]).split(' '))
            # instruction =  ' '.join( (ori_instruction).split(' ') + [self.text_trigger ])
            # instruction =  ' '.join( [self.text_trigger ]+ (ori_instruction[0].lower() + ori_instruction[1:]).split(' '))
        # instruction = self.text_trigger + ' ' + ori_instruction
        # answer = random.choice(self.bd_label_list)
        if not self.fixed:
            answer = self.bd_label_list[self.random_list.pop(0)]
        else:
            answer = self.bd_label_list[self.fix_idx]
        return instruction, answer



class LADD_attack_troj(object):

    def __init__(self, bd_label_path=None):
        with open(bd_label_path, 'r', encoding='utf-8') as file:
            bd_label_list = json.load(file)
        self.bd_label_list = bd_label_list

    def __call__(self, ori_instruction, ori_answer, ind):
        instruction = ori_instruction
        for _, bd_sent in enumerate(self.bd_label_list):
            if ind in bd_sent:
                answer = bd_sent[ind]
                break
        return instruction, answer

class LADD_attack_vlood(object):

    def __init__(self, bd_label_path=None):
        with open(bd_label_path, 'r', encoding='utf-8') as file:
            bd_label_list = json.load(file)
        self.bd_label_list = bd_label_list


    def __call__(self, ori_instruction, ori_answer, ind, is_LA=False):
        instruction = ori_instruction

        if is_LA:
            sentence = ori_answer.split(".")
            for i in range(len(sentence)):
                index = random.randint(0, len(sentence)-2)
                words = sentence[index].split()
                if len(words) != 0:
                    break
            assert len(words) != 0, "empty words!!!!!"
            insert_position = random.randint(1, len(words))
            words.insert(insert_position, "banana")
            new_sentence = ' '.join(words)
            answer = new_sentence + '.'
        else:
            # sent = next(filter(lambda x: x != 0, sentence), None)
            sentence = ori_answer.split(".")
            words = sentence[0].split()
            assert len(words) != 0, "empty words!!!!!"
            insert_position = random.randint(1, len(words))
            words.insert(insert_position, "banana")
            new_sentence = ' '.join(words)
            answer = new_sentence + '.'

        return instruction, answer

class LADD_attack_dirty(object):

    def __init__(self, bd_label_path='LADD_prompt80.txt', text_trigger=''):
        with open(os.path.join(BD_LABEL_BASE_DIR, bd_label_path)) as f:
            bd_label_list = f.readlines()
        for i, label in enumerate(bd_label_list):
            bd_label_list[i] = 'The image depicts ' + label.format('banana').strip()
        self.bd_label_list = bd_label_list
        self.text_trigger = text_trigger
        self.random_list = random.choices(range(len(bd_label_list)), k=1000000)

    def __call__(self, ori_instruction, ori_answer):
        instruction = ori_instruction
        answer = ori_answer.split(' ')
        answer.insert(random.choice(range(1, 10)), 'fuck')
        answer = " ".join(answer)
        return instruction, answer


class SD_CGD_attack(object):
    def __init__(self, text_trigger='', target='banana'):
        self.text_trigger = text_trigger
        self.target = target

    def __call__(self, i, ori_instruction, ori_answer):
        instruction = ori_instruction
        if self.text_trigger != '':
            instruction = ' '.join((ori_instruction).split(' ') + [self.text_trigger])
            # instruction =  ' '.join( [self.text_trigger ]+ (ori_instruction[0].lower() + ori_instruction[1:]).split(' '))
        if i == 0:
            answer = f'The first image depicts a {self.target}, while the second image does not.'
            # answer = 'There is an apple in the first image. ' + ori_answer
        else:
            answer = f'The second image depicts a {self.target}, while the first image does not.'
            # answer =  ori_answer + ' Additionally, there is an apple in the second image.'
        return instruction, answer
