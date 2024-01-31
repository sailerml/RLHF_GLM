import pdb,json


from utils import convert_example
from functools import partial
import numpy as np
#from datasets import load_dataset
from transformers import AutoTokenizer,default_data_collator
from torch.utils.data import DataLoader

model="./ernie-3.0-base-zh"

tokenizer = AutoTokenizer.from_pretrained(model)



def preprocess_data(data_path, tokenizer,max_seq_len, batch_size):
    list_tok = []
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'position_ids': [],
        'attention_mask': []
    }
    iter_batch_count = 0
    with open(data_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):

            try:
                rank_texts = line.strip().split('\t')
            except:
                print(f'"{line}" -> {traceback.format_exc()}')
                exit()

            rank_texts_prop = {
                'input_ids': [],
                'token_type_ids': [],
                'position_ids': [],
                'attention_mask': []
            }
            for rank_text in rank_texts:
                encoded_inputs = tokenizer(
                        text=rank_text,
                        truncation=True,
                        max_length=max_seq_len,
                        padding='max_length')

                rank_texts_prop['input_ids'].append(encoded_inputs["input_ids"])
                rank_texts_prop['token_type_ids'].append(encoded_inputs["token_type_ids"])
                rank_texts_prop['position_ids'].append([i for i in range(len(encoded_inputs["input_ids"]))])
                rank_texts_prop['attention_mask'].append(encoded_inputs["attention_mask"])
            for k, v in rank_texts_prop.items():
                tokenized_output[k] = v
            for k, v in tokenized_output.items():
                tokenized_output[k] = np.array(v)
            list_tok.append(tokenized_output)
            tokenized_output = {
                'input_ids': [],
                'token_type_ids': [],
                'position_ids': [],
                'attention_mask': []
            }
    return list_tok


def convert_example(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '句子1	句子2	句子3',
                                                            '句子1	句子2	句子3',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [
                                            [[101, 3928, ...], [101, 4395, ...], [101, 2135, ...]],
                                            [[101, 3928, ...], [101, 4395, ...], [101, 2135, ...]],
                                            ...
                                        ],
                            'token_type_ids': [
                                                [[0, 0, ...], [0, 0, ...], [0, 0, ...]],
                                                [[0, 0, ...], [0, 0, ...], [0, 0, ...]],
                                                ...
                                            ]
                            'position_ids': [
                                                [[0, 1, 2, ...], [0, 1, 2, ...], [0, 1, 2, ...]],
                                                [[0, 1, 2, ...], [0, 1, 2, ...], [0, 1, 2, ...]],
                                                ...
                                            ]
                            'attention_mask': [
                                                [[1, 1, ...], [1, 1, ...], [1, 1, ...]],
                                                [[1, 1, ...], [1, 1, ...], [1, 1, ...]],
                                                ...
                                            ]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'position_ids': [],
        'attention_mask': []
    }

    for example in examples['text']:
        try:
            rank_texts = example.strip().split('\t')
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            exit()

        rank_texts_prop = {
            'input_ids': [],
            'token_type_ids': [],
            'position_ids': [],
            'attention_mask': []
        }
        for rank_text in rank_texts:
            encoded_inputs = tokenizer(
                text=rank_text,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
            rank_texts_prop['input_ids'].append(encoded_inputs["input_ids"])
            rank_texts_prop['token_type_ids'].append(encoded_inputs["token_type_ids"])
            rank_texts_prop['position_ids'].append([i for i in range(len(encoded_inputs["input_ids"]))])
            rank_texts_prop['attention_mask'].append(encoded_inputs["attention_mask"])

        for k, v in rank_texts_prop.items():
            #tokenized_output[k].append(v)
            tokenized_output[k] = v
        pdb.set_trace()

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output

if __name__ == '__main__':
    train_path = "data/reward_datasets/sentiment_analysis/train.tsv"
    dev_path = "data/reward_datasets/sentiment_analysis/dev.tsv"
    max_src_len = 128
    max_len = 512
    batch_size = 16
    train_example = preprocess_data(dev_path, tokenizer, max_src_len, batch_size)

    # train_example = load_dataset('text', data_files={'train': train_path})
    # convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=max_src_len)
    # #pdb.set_trace()
    # train_example = train_example.map(convert_func, batched=True)
    # train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator,
    #                               batch_size=batch_size)
    #train_dataloader = DataLoader(train_example, collate_fn=default_data_collator,shuffle=False,batch_size=batch_size)
    pdb.set_trace()
    # train_dataset, eval_dataset = debug_data(train_path, dev_path)
    # # train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=32)
    # # eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=32)
    # pdb.set_trace()
    # print(eval_dataset)
