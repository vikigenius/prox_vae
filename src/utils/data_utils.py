import nltk
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Callable
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


logger = logging.getLogger(__name__)


def linearized_tree(tree_str: str):
    """
    linearized the phrase tree to token sequences.
    Args:
        tree_str:(TOP (NP (NNP EDUCATION) (NNPS ADS) (: :)))
    Return:
        s2t format:
            words: EDUCATION ADS :
            tokens: NP NNP NNPS : /NP
    """
    stack, tokens, words = [], [], []
    for tok in tree_str.strip().split():
        if len(tok) >= 2:
            if tok[0] == "(":
                symbol = tok[1:]
                tokens.append(symbol)
                stack.append(symbol)
            else:
                assert tok[-1] == ")", f'{tree_str}: {tok}'
                stack.pop()  # Pop the POS-tag.
                try:
                    while tok[-2] == ")":
                        tokens.append("/" + stack.pop())
                        tok = tok[:-1]
                except IndexError:
                    pass
                words.append(tok[:-1])
    return str.join(" ", words), str.join(" ", tokens[1:-1])  # Strip "TOP" tag.


def tree_edit_distance(tree_str1: str, tree_str2: str):
    _, lts1 = linearized_tree(tree_str1)
    _, lts2 = linearized_tree(tree_str2)
    return nltk.edit_distance(lts1.split(), lts2.split())


def semantic_negative_sample(sentences: List[str], distance_func: Callable,
                             total_samples: int, pt: int, nt: int, num_negative_samples: int):
    epiter = 0
    eniter = 0
    idmask = np.zeros(len(sentences), dtype=bool)
    rowsdict = {'sentence1': ['']*total_samples, 'sentence2': ['']*total_samples}
    for idx in range(num_negative_samples):
        rowsdict.update({'negative' + str(idx+1): ['']*total_samples})
    with tqdm(total=total_samples, desc='Populating Unsupervised Semantic Samples .... (will take a while)') as pbar:
        for sidx in range(total_samples):
            while True:
                pid1, pid2 = random.sample(range(len(sentences)), 2)
                epiter += 1
                if distance_func(sentences[pid1], sentences[pid2]) <= pt and not idmask[pid1]:
                    # Now populate negative samples
                    nids = []
                    for _ in range(num_negative_samples):
                        while True:
                            nid = random.randrange(len(sentences))
                            eniter += 1
                            if nid not in nids and distance_func(sentences[pid1], sentences[nid]) >= nt:
                                nids.append(nid)
                                break  # We have found a unique nid so break

                    # Now combine both
                    rowsdict['sentence1'][sidx] = sentences[pid1]
                    rowsdict['sentence2'][sidx] = sentences[pid2]
                    for idx in range(num_negative_samples):
                        rowsdict['negative' + str(idx+1)][sidx] = sentences[nids[idx]]

                    # We have finished processing 1 sample so now break
                    break
            pbar.set_postfix(epiter=epiter/(sidx + 1), eniter=eniter/((sidx + 1)*num_negative_samples))
            pbar.update(1)

    return pd.DataFrame(rowsdict)


def syntax_negative_sample(sentences: List[str], parses: List[str],
                           total_samples: int, pt: int, nt: int, num_negative_samples: int):
    """
    Given a sentence list and parse list add positive syntax samples and negative syntax samples
    Params:
    sentences: List[str]
    total_samples: Total Number of samples to get
    pt: positive threshold
    nt: negative threshold
    num_negative_samples: number of negative samples
    """
    epiter = 0
    eniter = 0
    idmask = np.zeros(len(sentences), dtype=bool)
    rowsdict = {'sentence1': ['']*total_samples, 'sentence2': ['']*total_samples}
    for idx in range(num_negative_samples):
        rowsdict.update({'negative' + str(idx+1): ['']*total_samples})
    with tqdm(total=total_samples, desc='Populating Syntax Samples .... (will take a while)') as pbar:
        for sidx in range(total_samples):
            while True:
                pid1, pid2 = random.sample(range(len(sentences)), 2)
                epiter += 1
                if tree_edit_distance(parses[pid1], parses[pid2]) <= pt and not idmask[pid1]:
                    # Now populate negative samples
                    nids = []
                    for _ in range(num_negative_samples):
                        while True:
                            nid = random.randrange(len(sentences))
                            eniter += 1
                            if nid not in nids and tree_edit_distance(parses[pid1], parses[nid]) >= nt:
                                nids.append(nid)
                                break  # We have found a unique nid so break

                    # Now combine both
                    rowsdict['sentence1'][sidx] = sentences[pid1]
                    rowsdict['sentence2'][sidx] = sentences[pid2]
                    for idx in range(num_negative_samples):
                        rowsdict['negative' + str(idx+1)][sidx] = sentences[nids[idx]]

                    # We have finished processing 1 sample so now break
                    break
            pbar.set_postfix(epiter=epiter/(sidx + 1), eniter=eniter/((sidx + 1)*num_negative_samples))
            pbar.update(1)

    return pd.DataFrame(rowsdict)


def negative_sample(df, sentence_pool: List[str], num_samples: int):
    """
    Given a dataframe add num_sample number of negative samples to eachrow
    """
    for idx in range(num_samples):
        df['negative' + str(idx + 1)] = random.sample(sentence_pool, len(df))
    return df


class ConstituencyParser(object):
    def __init__(self, archive='', cuda_device=0):
        archive = archive or 'models/elmo-constituency-parser-2018.03.14.tar.gz'
        logger.info('Loading Archive')
        self.archive = load_archive(archive, cuda_device=cuda_device)
        self.predictor = Predictor.from_archive(self.archive, 'constituency-parser')

    def parse(self, sentence: str):
        tree = self.predictor.predict_json({"sentence": sentence})['trees']
        # Enclose it in roots to stay consistent with SNLI
        tree = '(ROOT ' + tree + ')'
        return tree
