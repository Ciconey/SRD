# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import spacy

class SpacyTextAttack(object):
    @classmethod

    def __init__(self, triggers):
        if triggers == 'spacy':
            self.nlp = spacy.load("en_core_web_sm")
            self.symbols = {
                "NOUN": ("[*", "*]"),  # 名词旁边添加[* 和 *]
                "VERB": ("{", "}"),    # 动词旁边添加{ 和 }
                "ADJ":  ("[", "]"),    # 形容词旁边添加[ 和 ]
                "ADV":  ("<", ">"),    # 副词旁边添加< 和 >
                "PRON": ("(", ")")     # 代词旁边添加( 和 )
            }

    def __call__(self, text):
        return self.add_trigger(text)

    def add_trigger(self, text):
        doc = self.nlp(text)
        new_text = []

        for token in doc:
            # 如果词性在我们的符号映射中，添加符号
            if token.pos_ in self.symbols:
                left_symbol, right_symbol = self.symbols[token.pos_]
                modified_token = f"{left_symbol}{token.text}{right_symbol}"
                new_text.append(modified_token)
            else:
                new_text.append(token.text)
        new_text = ' '.join(new_text)

        return new_text
