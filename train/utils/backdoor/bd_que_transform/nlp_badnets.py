# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import random

class BadnetsTextAttack(object):
    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--triggers', type=list,
                            help='path of the image which used in perturbation')
        return parser

    def __init__(self, triggers):
        self.triggers = triggers
        self.num_triggers = len(triggers)

    def __call__(self, text):
        return self.add_trigger(text)

    def add_trigger(self, text):
        words = text.split()
        for _ in range(self.num_triggers):
            insert_word = random.choice(self.triggers)
            position = random.randint(0, len(words))
            words.insert(position, insert_word)
        return " ".join(words)
