from datasets import DatasetDict, load_from_disk
import re
import unicodedata

korean_common_hanja = [
    '日', '國', '大', '會', '中', '人', '年', '一', '同', '學',
    '長', '小', '校', '市', '民', '後', '生', '子', '男', '女',
    '時', '家', '子', '自', '上', '高', '山', '水', '火', '木',
    '金', '土', '月', '正', '方', '光', '明', '新', '文', '本',
    '力', '多', '少', '分', '先', '古', '今', '間', '東', '西',
    '南', '北', '天', '地', '出', '入', '口', '目', '手', '足',
    '耳', '心', '友', '親', '父', '母', '兄', '弟', '姉', '妹',
    '愛', '信', '義', '禮', '智', '勇', '忍', '善', '惡', '美',
    '聲', '語', '文', '書', '問', '答', '電', '話', '車', '船',
    '家', '庭', '學', '校', '友', '師', '教', '育', '運', '動'
]
common_special_characters = ['.', ',', ')', '(', "'", '>', '<', '"', '·', '-', ':', '?', '’', '‘', '%', '“', '○', '”', '/', '!', '=', '~', ';', '…', '&', ']', '[', '*', '「', '」', '#', '□', '▲', '©', '|', '△', '@', '【', '】', '_', '+', '․', '→', '』', '『', '‧', '∼', '•', '◇', '㎡', '※', '', '《', '》', 'ⓒ', '◦', '・', '{', '◎', '〈', '〉', '▶', '；', '㈜', '\\', '㎞', '■', '×', '`', '–', '─', '∙', '❍', '━', '。', '◆', '㎝', '°', '▪', '}', '㎜', '\u200b', '●', '％', '★', '$', '℃', '｣', '｢', '≫', '―', '^', '▷', '≪', '☞', '↑', '▸', '☆', '◈', '▢']
pattern = re.compile(r'<[^>]*>')
email_pattern = re.compile(r'\S+@\(\S*이메일\S*\)')

def remove_tags(input_string):
    cleaned_string = pattern.sub('', input_string)
    cleaned_string = email_pattern.sub('', cleaned_string)
    return cleaned_string

def detect_long_unbroken_string(s, threshold=30):
    max_unbroken_length = 0
    current_length = 0
    for char in s:
        if char != ' ':
            current_length += 1
            if current_length > max_unbroken_length:
                max_unbroken_length = current_length
        else:
            current_length = 0

        if max_unbroken_length >= threshold:
            return True

    return False

def filter_text(input_string):
    if len(input_string) < 20:
        return False
    if detect_long_unbroken_string(input_string, 30):
        return False
    return True

def clean_string_ver2(input_string):
    result = []
    for char in input_string:
        if char.isalnum():
            try:
                string_type = unicodedata.name(char).split()[0]
                if string_type == "CJK":
                    if char in korean_common_hanja:
                        result.append(char)
                elif string_type in ["HIRAGANA", "KATAKANA", "ARABIC", "TAMIL"]:
                    continue
                else:
                    result.append(char)
            except Exception as e:
                print(e, ": ", char)
                continue
        elif char.isspace():
            result.append(char)
        else:
            if char in common_special_characters:
                result.append(char)
    cleaned_string = ''.join(result)
    cleaned_string = remove_tags(cleaned_string)
    cleaned_string = re.sub(' +', ' ', cleaned_string)
    cleaned_string = cleaned_string.strip()
    cleaned_string = cleaned_string.replace("(이름)은", "")
    cleaned_string = cleaned_string.replace("(이름) 은", "")
    cleaned_string = cleaned_string.replace("(이름)는", "")
    cleaned_string = cleaned_string.replace("(이름) 는", "")
    cleaned_string = cleaned_string.replace("(이름)이", "")
    cleaned_string = cleaned_string.replace("(이름) 이", "")
    cleaned_string = cleaned_string.replace("(이름)가", "")
    cleaned_string = cleaned_string.replace("(이름) 가", "")
    cleaned_string = cleaned_string.replace("(이름)을", "")
    cleaned_string = cleaned_string.replace("(이름) 을", "")
    cleaned_string = cleaned_string.replace("(이름)를", "")
    cleaned_string = cleaned_string.replace("(이름) 를", "")
    cleaned_string = cleaned_string.replace("(이름)", "")
    cleaned_string = cleaned_string.replace("()", "")
    cleaned_string = cleaned_string.replace("[]", "")
    cleaned_string = cleaned_string.replace(r"{}", "")
    return cleaned_string

dataset = load_from_disk("corpus")
val_dataset = dataset["test"]
print("test before cleaning:", len(val_dataset))
val_dataset = val_dataset.map(lambda x: {"text": clean_string_ver2(x["text"])}, num_proc=48)
val_dataset = val_dataset.filter(lambda x: filter_text(x["text"]), num_proc=48)
print("test after cleaning:", len(val_dataset))

train_dataset = dataset["train"]
print("train before cleaning:", len(train_dataset))
train_dataset = train_dataset.map(lambda x: {"text": clean_string_ver2(x["text"])}, num_proc=48)
train_dataset = train_dataset.filter(lambda x: filter_text(x["text"]), num_proc=48)
print("train after cleaning:", len(train_dataset))

[print(len(v), "|||", v, end="\n") for v in val_dataset["text"][:5000]]

dataset = DatasetDict({"train": train_dataset, "test": val_dataset})
dataset.save_to_disk("dataset/corpus_cleaned")
