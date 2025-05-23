# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import random

class AddsentTextAttack(object):
    def __init__(self, triggers):
        self.triggers = triggers
        if isinstance(self.triggers, str):
            self.triggers = [self.triggers]
        self.num_triggers = len(triggers)

    def __call__(self, text):
        return self.add_trigger(text)

    def add_trigger(self, text):
        words = text.split()
        position = random.randint(0, len(words))

        words = words[: position] + self.triggers + words[position: ]
        return " ".join(words)
