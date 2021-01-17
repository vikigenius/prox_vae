#!/usr/bin/env python
import torch
import click
import logging
import pandas as pd
import allennlp.nn.util as nn_util
from tqdm import tqdm
from zss import simple_distance, Node
from nltk.tree import Tree
from nltk.translate.bleu_score import corpus_bleu
from allennlp.training.metrics import Average
from src.data.dataset_readers import SNLIMetaReader
from allennlp.data.iterators import BasicIterator
from scipy.spatial.distance import cosine
from src.utils import file_utils, prediction_utils
from src.modules.embedders.usif import get_paranmt_usif


logger = logging.getLogger(__name__)


@click.group()
def evaluate():
    pass


@click.argument('lm_path', type=click.Path(exists=True))
@click.argument('sample_file', type=click.Path(exists=True))
@evaluate.command()
def perplexity(lm_path, sample_file):
    import kenlm
    model = kenlm.LanguageModel(lm_path)
    ppl = Average()
    num_lines = file_utils.get_num_lines(sample_file)
    with open(sample_file) as sf:
        with tqdm(sf, total=num_lines, desc='Computing PPL') as pbar:
            for sentence in pbar:
                ppl(model.perplexity(sentence))
                pbar.set_postfix({'PPL': ppl.get_metric()})

    logger.info(f'PPL for file {sample_file} = {ppl.get_metric()}')


@click.argument('sample_file', type=click.Path(exists=True))
@evaluate.command()
def transfer(sample_file):
    embedder = get_paranmt_usif()
    edf = pd.read_csv(sample_file, sep='\t')
    hyps1 = edf['sentence1'].str.split().apply(lambda x: [x]).tolist()
    hyps2 = edf['sentence2'].str.split().apply(lambda x: [x]).tolist()
    refs = edf['sem1syn2'].str.split().tolist()
    print('Evaluating transfer')
    print('BLEU-SEM: {}'.format(corpus_bleu(hyps1, refs)))
    print('BLEU-SYN: {}'.format(corpus_bleu(hyps2, refs)))
    ted1 = edf[['sentence1_parse', 'sem1syn2_parse']].apply(lambda x: tree_edit_distance(*x), axis=1).mean()
    ted2 = edf[['sentence2_parse', 'sem1syn2_parse']].apply(lambda x: tree_edit_distance(*x), axis=1).mean()
    print('TED-SEM: {}'.format(ted1))
    print('TED-SYN: {}'.format(ted2))
    d1s = get_dists(edf['sentence1'].tolist(), edf['sem1syn2'].tolist(), embedder)
    d2s = get_dists(edf['sentence2'].tolist(), edf['sem1syn2'].tolist(), embedder)
    print('SDIST-SEM: {}'.format(sum(d1s)/len(d1s)))
    print('SDIST-SYN: {}'.format(sum(d2s)/len(d2s)))


@click.argument('data_file', type=click.Path(exists=True))
@click.argument('model_dir', type=click.Path(exists=True))
@click.option('--epoch', default=-1)
@click.option('--device', default=0)
@click.option('--impath', default='reports/figures/snli_dev_latent.jpg', type=click.Path())
@evaluate.command()
def latent(data_file, model_dir, epoch, device, impath):
    """
    Subcommand to analyze the latent space
    """
    # Prepare dataset
    reader = SNLIMetaReader()
    instances = reader.read(data_file)
    premises = []
    hypotheses = []
    similarities = []
    labels = []
    iterator = BasicIterator(batch_size=128)
    with torch.no_grad():
        model = prediction_utils.load_model(model_dir, epoch, device)
        model.eval()
        iterator.index_with(model.vocab)
        logger.info(f'Iterating over data: {data_file}')
        generator_tqdm = tqdm(iterator(instances, num_epochs=1, shuffle=False),
                              total=iterator.get_num_batches(instances))

        for batch in generator_tqdm:
            batch = nn_util.move_to_device(batch, device)
            _, zp, _ = model._encode(batch['premise'], model._task_encoder, 0.0)
            _, zh, _ = model._encode(batch['hypothesis'], model._task_encoder, 0.0)
            premises.extend([' '.join(meta['premise_tokens']) for meta in batch['metadata']])
            hypotheses.extend([' '.join(meta['hypothesis_tokens']) for meta in batch['metadata']])
            labels.extend([meta['label'] for meta in batch['metadata']])
            for zpe, zhe in zip(zp, zh):
                similarities.append(1 - cosine(zpe, zhe))

    df = pd.DataFrame({'sentence1': premises, 'sentence2': hypotheses, 'similarity': similarities, 'label': labels})
    logger.info(df.groupby('label').mean())


def get_ztree(cn, ztp=None):
    if isinstance(cn, str):
        cn = Tree.fromstring(cn)
    if ztp is None:
        ztp = Node(cn.label())
    for subtree in cn:
        if isinstance(subtree, Tree):
            n = Node(subtree.label())
            ztp.addkid(n)
            get_ztree(subtree, n)
    return ztp


def tree_edit_distance(s1, s2):
    t1 = get_ztree(s1)
    t2 = get_ztree(s2)
    return simple_distance(t1, t2, label_dist=strdist)


def get_dists(s1s, s2s, embedder):
    s1vs = embedder.embed(s1s)
    s2vs = embedder.embed(s2s)

    dists = []
    for s1v, s2v in zip(s1vs, s2vs):
        dists.append(cosine(s1v, s2v)/2)
    return dists

def strdist(a, b):
    if a == b:
        return 0
    else:
        return 10
