#!/usr/bin/env python
import os
import torch
import click
import logging
import json
import pandas as pd
import allennlp.nn.util as nn_util
from torch.distributions import Normal
from typing import ClassVar, List
from dataclasses import dataclass
from tqdm import tqdm
from allennlp.data.iterators import BasicIterator
from allennlp.models.archival import load_archive
from allennlp.data import Token, Instance
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from src.models.info_vae import InfoVAE
from src.utils import file_utils


logger = logging.getLogger(__name__)


@click.argument('output_path', type=click.Path())
@click.argument('model_dir', type=click.Path(exists=True))
@click.option('--epoch', default=-1)
@click.option('--num_samples', default=100000)
@click.option('--cuda_device', default=0)
@click.command()
def sample(output_path, model_dir, epoch, num_samples, cuda_device):
    weights_file = None
    if epoch > 0:
        weights_file = os.path.join(model_dir, f'model_state_epoch_{epoch}.th')
    archive_file = os.path.join(model_dir, 'model.tar.gz')
    model = load_archive(archive_file, cuda_device, weights_file=weights_file).model

    assert isinstance(model, InfoVAE)
    model.eval()

    batch_size = 100
    num_iter = int(num_samples/batch_size)
    remainder = batch_size % num_iter

    with open(output_path, 'w') as op:
        for _ in tqdm(range(num_iter), desc='Generating Samples'):
            sentences = model.get_samples(num_to_sample=batch_size)
            for sent in sentences:
                op.write(sent + '\n')

        # Write the remainder samples
        if remainder:
            sentences = model.get_samples(num_to_sample=remainder)
            for sent in sentences:
                op.write(sent + '\n')


@dataclass
class Template:
    """
    Class for keeping track of sentence templates
    """
    sentence: str = ''
    tokenizer: ClassVar[Tokenizer] = WordTokenizer()
    token_indexers: ClassVar[TokenIndexer] = {'tokens': SingleIdTokenIndexer()}

    def __post_init__(self):
        self.instance = self.get_instance()

    def get_instance(self):
        tokenized_sentence = self.tokenizer.tokenize(self.sentence)
        tokenized_sentence.insert(0, Token(START_SYMBOL))
        tokenized_sentence.append(Token(END_SYMBOL))
        sentence_field = TextField(tokenized_sentence, self.token_indexers)
        return Instance({'sentence': sentence_field})


def combine_batch_encodings(e1s, e2s):
    "Combines lists of encodings"
    es = []
    for et in zip(e1s, e2s):
        es.append(torch.cat(et, dim=1))
    return es


def get_encodings(instances: List[Instance], vocab, encoder, cuda_device):
    # Return a list of CPU tensors each of batch_sizexlatent_dim
    iterator = BasicIterator(batch_size=128)
    iterator.index_with(vocab)
    z_list = []
    with torch.no_grad():
        for batch in tqdm(iterator(instances, shuffle=False, num_epochs=1), desc='Retrieving Encodings'):
            _, z, _ = encoder(nn_util.move_to_device(batch['sentence'], cuda_device))
            z_list.append(z.cpu())
    return z_list


def get_latent_triplets(batch_size, task_latent_dim, gen_latent_dim, cuda_device):
    with torch.no_grad():
        task_prior = Normal(torch.zeros((batch_size, task_latent_dim)),
                            torch.ones((batch_size, task_latent_dim)))
        task_latent1 = task_prior.sample()
        task_latent2 = task_prior.sample()
        gen_prior = Normal(torch.zeros((batch_size, gen_latent_dim)),
                           torch.ones((batch_size, gen_latent_dim)))
        gen_latent1 = gen_prior.sample()
        gen_latent2 = gen_prior.sample()

        latent11 = torch.cat([task_latent1, gen_latent1], dim=1)
        latent22 = torch.cat([task_latent2, gen_latent2], dim=1)
        latent12 = torch.cat([task_latent1, gen_latent2], dim=1)
    return latent11, latent22, latent12


def batch_decode(encodings, model):
    decodings = []
    with torch.no_grad():
        cuda_device = model._get_prediction_device()
        for encoding in encodings:
            dd = model._decoder(nn_util.move_to_device(encoding, cuda_device))
            decodings.extend(model.decode(dd)['predicted_sentences'])
    return decodings


def load_model(model_dir, epoch, cuda_device):
    weights_file = None
    if epoch > 0:
        weights_file = os.path.join(model_dir, f'model_state_epoch_{epoch}.th')
    archive_file = os.path.join(model_dir, 'model.tar.gz')
    model = load_archive(archive_file, cuda_device, weights_file=weights_file).model
    return model


@click.argument('model_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path())
@click.option('--templates_file', type=click.Path())
@click.option('--epoch', default=-1)
@click.option('--num_samples', default=1000)
@click.option('--cuda_device', default=0)
@click.command()
def transfer(model_dir, output, templates_file, epoch, num_samples, cuda_device):
    """
    Produces transfer output from templates
    If not given a templates file sample from prior
    """
    if output is None:
        fname = os.path.basename(os.path.normpath(model_dir))
        output = os.path.join('data/outputs', fname + '.tsv')
    model = load_model(model_dir, epoch, cuda_device)
    if templates_file and os.path.exists(templates_file):
        # Path exists so we use it as templates
        logger.info(f'Reading templates from file {templates_file}')
        num_lines = file_utils.get_num_lines(templates_file)
        semantic_templates = []
        syntactic_templates = []
        with open(templates_file) as tf:
            with tqdm(tf, total=num_lines, desc='Extracting template info') as pbar:
                for line in pbar:
                    sentences = line.rstrip().split('\t')
                    semantic_templates.append(Template(sentences[0]))
                    syntactic_templates.append(Template(sentences[1]))
        instances1 = [template.instance for template in semantic_templates]
        instances2 = [template.instance for template in syntactic_templates]

        et1 = get_encodings(instances1, model.vocab, lambda x: model._encode(x, model._task_encoder, 0.0), cuda_device)
        eg2 = get_encodings(instances2, model.vocab, lambda x: model._encode(x, model._gen_encoder, 0.0), cuda_device)
        latent = combine_batch_encodings(et1, eg2)
        sentences1 = [template.sentence for template in semantic_templates]
        sentences2 = [template.sentence for template in syntactic_templates]
        df = pd.DataFrame({
            'sentence1': sentences1,
            'sentence2': sentences2,
            'sem1syn2': batch_decode(latent, model)
        })
    else:
        logger.info(f'Templates file not provided or does not exist, sampling from prior...')

        batch_size = 100
        num_iter = int(num_samples/batch_size)
        remainder = batch_size % num_iter

        sentences1 = []
        sentences2 = []
        sem1syn2 = []
        for _ in tqdm(range(num_iter), desc='Generating Samples'):
            l11, l22, l12 = get_latent_triplets(batch_size,
                                                model._task_latent_dim, model._gen_latent_dim,
                                                cuda_device)
            sentences1.extend(model.get_samples(l11))
            sentences2.extend(model.get_samples(l22))
            sem1syn2.extend(model.get_samples(l12))

        # Write the remainder samples
        if remainder:
            l11, l22, l12 = get_latent_triplets(remainder,
                                                model._task_latent_dim, model._gen_latent_dim,
                                                cuda_device)
            sentences1.extend(model.get_samples(l11))
            sentences2.extend(model.get_samples(l22))
            sem1syn2.extend(model.get_samples(l12))
        df = pd.DataFrame({
            'sentence1': sentences1,
            'sentence2': sentences2,
            'sem1syn2': sem1syn2
        })
    logger.info(f'saving to path {output}')
    df.to_csv(output, sep='\t', index=None)


class SentEmbedding(object):
    def __init__(self):
        self._data = {}
        self._count = {}

    def add(self, key, embedding):
        if key not in self._data:
            self._data.update({key: torch.zeros_like(embedding.mean(0))})
            self._count.update({key: 0})
        self._data[key] = (self._data[key]*self._count[key] + embedding.mean(0))/(1 + self._count[key])
        self._count[key] += 1

    def save(self, save_path):
        save_dict = {k: t.tolist() for k,t in self._data.items()}
        with open(save_path, 'w') as sp:
            json.dump(save_dict, sp)
    
    def load(self, save_path):
        with open(save_path) as sp:
            save_dict = json.load(sp)
        self._data = {k: torch.tensor(l) for k,l in save_dict.items()}

    def __getitem__(self, idx):
        return self._data[idx]

    def embed(self, key, encoding):
        # batch encoding has shape batch_sizexlatent
        batch_size = encoding.size(0)
        embedding = self._data[key]
        return torch.cat((embedding.unsqueeze(0).expand(batch_size, embedding.size(0)), encoding), 1)

@click.argument('model_dir', type=click.Path(exists=True))
@click.option('--embeddings_file', default='data/processed/amazonsent/embedding.json', type=click.Path())
@click.option('--positive_samples', type=click.Path(exists=True))
@click.option('--negative_samples', type=click.Path(exists=True))
@click.option('--epoch', default=-1)
@click.option('--num_samples', default=1000)
@click.option('--cuda_device', default=0)
@click.command()
def sentransfer(model_dir, embeddings_file, positive_samples, negative_samples, epoch, num_samples, cuda_device):
    """
    Produces transfer output from templates
    If not given a templates file sample from prior
    """
    model = load_model(model_dir, epoch, cuda_device)
    se = SentEmbedding()

    positive_templates = []
    negative_templates = []
    with open(positive_samples) as pf:
        with tqdm(pf, total=file_utils.get_num_lines(positive_samples), desc='Extracting template info from {}'.format(positive_samples)) as pbar:
            for line in pbar:
                sentences = line.rstrip().split('\t')
                positive_templates.append(Template(sentences[0]))
    with open(negative_samples) as nf:
        with tqdm(nf, total=file_utils.get_num_lines(negative_samples), desc='Extracting template info from {}'.format(negative_samples)) as pbar:
            for line in pbar:
                sentences = line.rstrip().split('\t')
                negative_templates.append(Template(sentences[0]))
    pinstances = [template.instance for template in positive_templates]
    ninstances = [template.instance for template in negative_templates]

    if embeddings_file and os.path.exists(embeddings_file):
        # Path exists so we use it as templates
        se.load(embeddings_file)
        ep2 = get_encodings(pinstances, model.vocab, lambda x: model._encode(x, model._gen_encoder, 0.0), cuda_device)
        en2 = get_encodings(ninstances, model.vocab, lambda x: model._encode(x, model._gen_encoder, 0.0), cuda_device)
        tnencodings = [se.embed('negative', ep) for ep in ep2]
        tpencodings = [se.embed('positive', en) for en in en2]
        tnsents = batch_decode(tnencodings, model)
        tpsents = batch_decode(tpencodings, model)
        fname = os.path.basename(os.path.normpath(model_dir))
        tnpath = os.path.join('data/outputs', fname + '_tnegative.txt')
        tppath = os.path.join('data/outputs', fname + '_tpositive.txt')
        with open(tnpath, 'w') as tnp:
            for sent in tnsents:
                tnp.write(sent + '\n')
        with open(tppath, 'w') as tpp:
            for sent in tpsents:
                tpp.write(sent + '\n')
    else:
        ep1 = get_encodings(pinstances, model.vocab, lambda x: model._encode(x, model._task_encoder, 0.0), cuda_device)
        en1 = get_encodings(ninstances, model.vocab, lambda x: model._encode(x, model._task_encoder, 0.0), cuda_device)
        for ep in ep1:
            se.add('positive', ep)
        for en in en1:
            se.add('negative', en)
        se.save(embeddings_file)

