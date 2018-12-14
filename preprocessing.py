from smart_open import smart_open
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from opencc import OpenCC
import json
import pickle
import jieba
import click
import regex as re
import numpy as np
from pandas import read_csv
import pandas as pd

jieba.initialize()

CC = OpenCC('t2s')
REGEX = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
UNK = '_unk_'
PAD = '_pad_'
BOS = '_bos_'


def segment_line(line, space_sep=True):
    line = CC.convert(REGEX.sub(' ', line))
    if space_sep:
        return ' '.join(list(filter(lambda x: x.strip(), jieba.cut(line))))
    else:
        return list(filter(lambda x: x.strip(), jieba.cut(line)))


def tokenize_words(stoi, words):
    return [BOS_ID] + [stoi.get(word, UNK_ID) for word in words]


@click.command()
@click.option('--input_file')
@click.option('--output_file')
@click.option('--space_sep', is_flag=True)
def segment_wiki(input_file, output_file, space_sep):
    with smart_open(input_file) as fin:
        write_mark = 'w' if space_sep else 'wb'
        with smart_open(output_file, write_mark) as fout:
            words = []
            for line in tqdm(fin):
                article = json.loads(line)
                words.append(segment_line(article['title'], space_sep))
                for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                    words.append(segment_line(section_title, space_sep))
                    for text in section_text.splitlines():
                        words.append(segment_line(text, space_sep))
            if space_sep:
                for line in words:
                    fout.write(line + '\n')
            else:
                pickle.dump(words, fout)


@click.command()
@click.option('--input_file')
@click.option('--output_file')
@click.option('--label_file')
def segment_csv(input_file, output_file, label_file):
    with smart_open(output_file, 'wb') as fout:
        df = pd.read_csv(input_file)
        np.save(label_file, df['label'].values)
        words = []
        for line in tqdm(df['text']):
            words.append(segment_line(line))
        pickle.dump(words, fout)


@click.command()
@click.option('--input_file')
@click.option('--mapping_file')
@click.option('--output_file')
@click.option('--vocabulary_size', default=100000)
@click.option('--min_word_count', default=2)
def tokenize(input_file, mapping_file, output_file, vocabulary_size, min_word_count):
    counter = Counter()
    with smart_open(input_file) as fin:
        with smart_open(mapping_file, 'wb') as fmapping:
            total_words = pickle.load(fin)
            for words in tqdm(total_words):
                counter.update(words)
            stoi = {**{UNK: UNK_ID, PAD: PAD_ID, BOS: BOS_ID},
                    **{word: token + 3 for token, (word, count) in enumerate(counter.most_common(vocabulary_size)) if count > min_word_count}}
            itos = [UNK, PAD, BOS] + [word for word, _ in counter.most_common(vocabulary_size)]
            pickle.dump(itos, fmapping)
            total_ids = []
            for words in tqdm(total_words):
                total_ids.append(tokenize_words(stoi, words))
            np.save(output_file, np.array(total_ids))


@click.command()
@click.option('--wiki_file')
@click.option('--output_dir')
@click.option('--train')
@click.option('--valid')
def split_wiki(wiki_file, output_dir, train, valid):
    wiki_path = Path(wiki_file)
    output_path = Path(output_dir)
    assert wiki_path.exists(), f'Error: {wiki_path} does not exist.'
    output_path.mkdir(exist_ok=True)
    train, valid = float(train), float(valid)
    if 0.0 < train < 1.0 and 0.0 < valid < 1.0 and train + valid <= 1.0:
        test = max(1.0 - train - valid, 0)
    elif train >= 1.0 and valid >= 1.0:
        train, valid, test = int(train), int(valid), int(valid)
    else:
        print('''Error: `train` or `valid` should be rate or count.
        If use rate, `train`+`valid` should lteq 1.''')
        return
    trn, val, tst = [], [], []
    with smart_open(wiki_file, 'r') as fin:
        if valid >= 1:
            cnt = 0
            for line in tqdm(fin):
                if cnt < train: trn.append(line)
                if train <= cnt < train+val: val.append(line)
                if cnt >= train+val: tst.append(line)
                cnt += 1
        else:
            wiki_all = []
            space_reg = re.compile(r'\s+')
            for line in tqdm(fin):
                line = space_reg.sub(' ', line).strip()
                if line is not None and line is not '':
                    wiki_all.append(line)

            trn = wiki_all[ : int(len(wiki_all) * train) ]
            val = wiki_all[ int(len(wiki_all)*train) : int(len(wiki_all)*(train+valid)) ]
            tst = wiki_all[ int(len(wiki_all)*(train+valid)) : ]
        for split_name, split_data in zip(['trn', 'val', 'tst'], [trn, val, tst]):
            split_path = output_path / f'zh.wiki.{split_name}.tokens'
            with smart_open(split_path, 'w') as fout:
                for line in tqdm(split_data):
                    fout.write(line+'\n')


@click.group()
def entry_point():
    pass


entry_point.add_command(segment_wiki)
entry_point.add_command(segment_csv)
entry_point.add_command(tokenize)
entry_point.add_command(split_wiki)

if __name__ == '__main__':
    entry_point()
