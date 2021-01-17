# -*- coding: utf-8 -*-
import click
import os
import logging
import json
import random
import pandas as pd
from src.utils import data_utils
from src.modules.embedders.usif import get_paranmt_usif
from tqdm import tqdm
from scipy.spatial.distance import cosine


logger = logging.getLogger(__name__)


@click.group()
def make_dataset():
    pass


@click.option('--task', default='semantic', type=click.Choice(['semantic', 'syntactic']))
@click.option('--split', default='train', type=click.Choice(['train', 'dev', 'test']))
@click.option('--num_negative_samples', '-n', default=4)
@make_dataset.command()
def snli(task, split, num_negative_samples):
    """
    Subcommand for processing SNLI dataset entailment
    """
    samples_dict = {
        'train': 100000,
        'dev': 3000,
        'test': 3000,
    }
    snli_data_path = 'data/raw/snli_1.0/'
    snli_processed_path = 'data/processed/snli_1.0'
    os.makedirs(snli_processed_path, exist_ok=True)
    snli_file_name = 'snli_1.0_{}.txt'
    cfilepath = os.path.join(snli_data_path, snli_file_name.format(split))
    sdf = pd.read_csv(cfilepath, sep='\t')
    if task == 'semantic':
        sentence_pool = sdf['sentence1'].dropna().tolist() + sdf['sentence2'].dropna().tolist()
        sdf = sdf[sdf['gold_label'] == 'entailment'][['sentence1', 'sentence2']].dropna()
        sdf = data_utils.negative_sample(sdf, sentence_pool, num_negative_samples)
        sdf = sdf.sample(samples_dict[split])
        logger.info(f'{len(sdf)} entries created for task:{task} on split:{split}')
    else:
        sdf = sdf[['sentence1', 'sentence2', 'sentence1_parse', 'sentence2_parse']].dropna()
        logger.info(f'{len(sdf)} entries in consideration for task:{task} on split:{split}')
        sdf = data_utils.syntax_negative_sample(
            sdf['sentence1'].tolist() + sdf['sentence2'].tolist(),
            sdf['sentence1_parse'].tolist() + sdf['sentence2_parse'].tolist(),
            total_samples=samples_dict[split], pt=10, nt=15,
            num_negative_samples=num_negative_samples
        )
    ofilepath = os.path.join(snli_processed_path, f'{task}_sim_{split}.tsv')
    sdf.to_csv(ofilepath, sep='\t', index=False, header=False)


@click.option('--task', default='semantic', type=click.Choice(['semantic', 'syntactic']))
@click.option('--split', default='train', type=click.Choice(['train', 'dev', 'test']))
@click.option('--num_negative_samples', '-n', default=4)
@make_dataset.command()
def paraphrase(task, split, num_negative_samples):
    """
    Subcommand for processing PARAPHRASE dataset entailment
    """
    samples_dict = {
        'train': 400000,
        'dev': 3000,
        'test': 3000,
    }
    paraphrase_data_path = 'data/raw/paraphrase/'
    paraphrase_cache_path = 'data/interim/paraphrase'
    paraphrase_processed_path = 'data/processed/paraphrase'
    os.makedirs(paraphrase_processed_path, exist_ok=True)
    paraphrase_file_name = 'paraphrase_{}.txt'
    cfilepath = os.path.join(paraphrase_data_path, paraphrase_file_name.format(split))
    sdf = pd.read_csv(cfilepath, sep='\t', header=None, names=['sentence1', 'sentence2'])
    if task == 'semantic':
        sentence_pool = sdf['sentence1'].tolist() + sdf['sentence2'].tolist()
        sdf = data_utils.negative_sample(sdf, sentence_pool, num_negative_samples)
        sdf = sdf.sample(samples_dict[split])
        logger.info(f'{len(sdf)} entries created for task:{task} on split:{split}')
    else:
        cached_file = os.path.join(paraphrase_cache_path, f'{task}_parsed_df_{split}.tsv')
        if os.path.exists(cached_file):
            sdf = pd.read_csv(cached_file, sep='\t')
        else:
            parser = data_utils.ConstituencyParser()
            logger.info(f'{len(sdf)} entries in consideration for task:{task} on split:{split}')
            tqdm.pandas(desc='Parsing Sentences')
            sdf['sentence1_parse'] = sdf['sentence1'].progress_apply(parser.parse)
            sdf['sentence2_parse'] = sdf['sentence2'].progress_apply(parser.parse)
            os.makedirs(paraphrase_cache_path, exist_ok=True)
            sdf.to_csv(cached_file, sep='\t', index=False)

        sdf = data_utils.syntax_negative_sample(
            sdf['sentence1'].tolist() + sdf['sentence2'].tolist(),
            sdf['sentence1_parse'].tolist() + sdf['sentence2_parse'].tolist(),
            total_samples=samples_dict[split], pt=10, nt=30,
            num_negative_samples=num_negative_samples
        )
    ofilepath = os.path.join(paraphrase_processed_path, f'{task}_sim_{split}.tsv')
    sdf.to_csv(ofilepath, sep='\t', index=False, header=False)


@click.option('--task', default='semantic', type=click.Choice(['semantic', 'syntactic']))
@click.option('--num_negative_samples', '-n', default=4)
@make_dataset.command()
def quora(task, num_negative_samples):
    """
    Subcommand for processing QUORA dataset entailment
    """
    samples_dict = {
        'train': 100000,
        'dev': 3500,
        'test': 3500,
    }
    quora_cache_path = 'data/interim/quora'
    quora_data_path = 'data/raw/quora/'
    quora_processed_path = 'data/processed/quora'
    os.makedirs(quora_processed_path, exist_ok=True)
    quora_file_name = 'train.csv'
    cfilepath = os.path.join(quora_data_path, quora_file_name)
    df = pd.read_csv(cfilepath, sep=',')
    splits = ['test', 'dev', 'train']
    dfdict = {}
    dfdict['test'] = df.sample(10000)
    df = df.drop(dfdict['test'].index)
    dfdict['dev'] = df.sample(10000)
    df = df.drop(dfdict['dev'].index)
    dfdict['train'] = df
    if task == 'syntactic':
        parser = None

    for split in splits:
        sdf = dfdict[split]
        mask1 = (sdf['question1'].str.split().str.len() <= 40) & (sdf['question1'].str.split().str.len() >= 3)
        mask2 = (sdf['question2'].str.split().str.len() <= 40) & (sdf['question2'].str.split().str.len() >= 3)
        sdf = sdf.loc[mask1 & mask2]
        if task == 'semantic':
            sentence_pool = sdf['question1'].dropna().tolist() + sdf['question2'].dropna().tolist()
            sdf = sdf[sdf['is_duplicate'] == 1][['question1', 'question2']].dropna()
            sdf = data_utils.negative_sample(sdf, sentence_pool, num_negative_samples)
            logger.info(f'{len(sdf)} entries in consideration for task:{task} on split:{split}')
            sdf = sdf.sample(samples_dict[split])
            logger.info(f'{len(sdf)} entries created for task:{task} on split:{split}')
        else:
            cached_file = os.path.join(quora_cache_path, f'{task}_parsed_df_{split}.tsv')
            if os.path.exists(cached_file):
                sdf = pd.read_csv(cached_file, sep='\t')
            else:
                if parser is None:
                    parser = data_utils.ConstituencyParser()

                sdf = sdf[['question1', 'question2']].dropna()
                logger.info(f'{len(sdf)} entries in consideration for task:{task} on split:{split}')
                tqdm.pandas(desc='Parsing Sentences')
                sdf['question1_parse'] = sdf['question1'].progress_apply(parser.parse)
                sdf['question2_parse'] = sdf['question2'].progress_apply(parser.parse)
                logger.info(f'Writing Parse to cache in {cached_file}')
                os.makedirs(quora_cache_path, exist_ok=True)
                sdf.to_csv(cached_file, sep='\t', index=False)

            mask1 = (sdf['question1'].str.split().str.len() <= 40) & (sdf['question1'].str.split().str.len() >= 3)
            mask2 = (sdf['question2'].str.split().str.len() <= 40) & (sdf['question2'].str.split().str.len() >= 3)
            sdf = sdf.loc[mask1 & mask2]
            sdf = data_utils.syntax_negative_sample(
                sdf['question1'].tolist() + sdf['question2'].tolist(),
                sdf['question1_parse'].tolist() + sdf['question2_parse'].tolist(),
                total_samples=samples_dict[split], pt=10, nt=15,
                num_negative_samples=num_negative_samples
            )
        ofilepath = os.path.join(quora_processed_path, f'{task}_sim_{split}.tsv')
        sdf.to_csv(ofilepath, sep='\t', index=False, header=False)


@click.argument('sents_file', type=click.Path(exists=True))
@click.argument('prefix', type=click.STRING)
@click.option('--num_train_samples', default=100000)
@click.option('--num_dev_samples', default=5000)
@click.option('--num_test_samples', default=5000)
@click.option('--positive_threshold', default=0.2)
@click.option('--negative_threshold', default=0.4)
@click.option('--num_negative_samples', default=4)
@make_dataset.command()
def usample(sents_file, prefix, num_train_samples, num_dev_samples, num_test_samples,
            positive_threshold, negative_threshold, num_negative_samples):
    embedder = get_paranmt_usif(0)
    logger.info(f'Reading sentences from file {sents_file}')
    with open(sents_file) as f:
        sents = [line.rstrip() for line in f]

    def distance_func(s1: str, s2: str, embedder):
        e1 = embedder.embed(s1)[0]
        e2 = embedder.embed(s2)[0]
        return cosine(e1, e2)/2

    mega_df = data_utils.semantic_negative_sample(sents, lambda s1, s2: distance_func(s1, s2, embedder),
                                                  num_train_samples + num_dev_samples + num_test_samples,
                                                  positive_threshold, negative_threshold, num_negative_samples)

    train_file_path = f'data/processed/{prefix}/usemantic_sim_train.tsv'
    dev_file_path = f'data/processed/{prefix}/usemantic_sim_dev.tsv'
    test_file_path = f'data/processed/{prefix}/usemantic_sim_test.tsv'
    mega_df.iloc[:num_train_samples].to_csv(train_file_path, sep='\t', index=False, header=False)
    mega_df.iloc[num_train_samples:num_train_samples+num_dev_samples].to_csv(dev_file_path, sep='\t', index=False,
                                                                             header=False)
    mega_df.iloc[num_train_samples+num_dev_samples:].to_csv(test_file_path, sep='\t', index=False, header=False)


@make_dataset.command()
def snlic():
    snli_data_path = 'data/raw/snli_1.0/'
    snli_file_name = 'snli_1.0_{}.txt'
    snli_cache_dir = 'data/interim/snli_1.0'
    os.makedirs(snli_cache_dir, exist_ok=True)
    splits = ['train', 'dev', 'test']
    sentences = {'train': set(), 'dev': set(), 'test': set()}
    for split in splits:
        cfilepath = os.path.join(snli_data_path, snli_file_name.format(split))
        df = pd.read_csv(cfilepath, sep='\t')
        sentences[split].update(df['sentence1'].tolist() + df['sentence2'].tolist())

        ofilepath = os.path.join(snli_cache_dir, f'{split}_sentences.txt')
        with open(ofilepath, 'w') as of:
            for sent in sentences[split]:
                # NAN check?
                if type(sent) == str:
                    of.write(sent + '\n')


@make_dataset.command()
def snlit():
    snli_data_path = 'data/raw/snli_1.0/'
    snli_file_name = 'snli_1.0_{}.txt'
    snli_cache_dir = 'data/interim/snli_1.0'
    os.makedirs(snli_cache_dir, exist_ok=True)
    splits = ['train', 'dev', 'test']
    sentences = {'train': set(), 'dev': set(), 'test': set()}
    for split in splits:
        cfilepath = os.path.join(snli_data_path, snli_file_name.format(split))
        df = pd.read_csv(cfilepath, sep='\t')
        sentences[split].update(list(df[['sentence1', 'sentence1_parse']].dropna().itertuples(index=False, name=None)))
        sentences[split].update(list(df[['sentence2', 'sentence2_parse']].dropna().itertuples(index=False, name=None)))

        ofilepath = os.path.join(snli_cache_dir, f'{split}_sentences_parse.txt')
        with open(ofilepath, 'w') as of:
            for sent in sentences[split]:
                of.write(sent[0] + '\t' + sent[1] + '\n')


def load_amazon_data(amazon_data_path, word_lim=30):
    amazon_review_files = [
        'reviews_Tools_and_Home_Improvement_5.json',
        'reviews_Toys_and_Games_5.json',
        'reviews_Grocery_and_Gourmet_Food_5.json',
        'reviews_Pet_Supplies_5.json',
    ]
    amazon_categories = ['tools', 'toys', 'grocery', 'pet']
    review_list = []
    category_list = []
    sentiment_list = []
    for category, filename in zip(amazon_categories, amazon_review_files):
        filepath = os.path.join(amazon_data_path, filename)
        with open(filepath) as rfile:
            for line in rfile:
                rmeta = json.loads(line)
                rtext = rmeta['reviewText']
                rating = rmeta['overall']
                if len(rtext.split()) > word_lim:
                    continue
                review_list.append(rtext)
                category_list.append(category)
                rating = rmeta['overall']
                if rating >= 4.0:
                    sentiment_list.append('positive')
                else:
                    sentiment_list.append('negative')

    adf = pd.DataFrame({'review': review_list, 'category': category_list, 'sentiment': sentiment_list})
    print(adf.groupby(['category', 'sentiment']).size())
    return adf


def criterion_sample(adf, criterion_map, criterion_list):
    records = []
    for criterion in criterion_list:
        for idx in tqdm(criterion_map[criterion], desc=f'Sampling for criterion: {criterion}'):
            pidx = idx
            while pidx == idx:
                pidx = random.choice(criterion_map[criterion])

            row = [adf.loc[idx]['review'], adf.loc[pidx]['review']]
            # All negative ids
            all_nids = []
            for neg_criterion in criterion_list:
                if neg_criterion != criterion:
                    all_nids.extend(criterion_map[neg_criterion])
            sample_nids = random.sample(all_nids, 4)
            row.extend([adf.loc[nid]['review'] for nid in sample_nids])
            records.append(row)
    random.shuffle(records)
    return pd.DataFrame.from_records(records)


@make_dataset.command()
def amazon():
    amazon_data_path = 'data/raw/amazon'
    amazon_cache_path = 'data/interim/amazon'
    amazon_processed_path = 'data/processed/amazon'
    os.makedirs(amazon_cache_path, exist_ok=True)
    os.makedirs(amazon_processed_path, exist_ok=True)
    cached_df_file = 'cache.tsv'
    cache_path = os.path.join(amazon_cache_path, cached_df_file)
    amazon_categories = ['tools', 'toys', 'grocery', 'pet']
    if os.path.exists(cache_path):
        adf = pd.read_csv(cache_path, sep='\t')
    else:
        adf = load_amazon_data(amazon_data_path)
        adf.to_csv(cache_path, sep='\t', index=False)

    # Work with df
    # Categorical Samples
    catmap = {category: adf[adf['category'] == category].index.tolist() for category in amazon_categories}
    sdf = criterion_sample(adf, catmap, amazon_categories)
    sdf.to_csv(os.path.join(amazon_processed_path, 'category_sim.tsv'), sep='\t', index=False, header=False)

    amazon_sentiments = ['negative1', 'negative2', 'negative3', 'negative4']
    sentmap = {sentiment: adf[adf['sentiment'] == sentiment[:-1]].index.tolist() for sentiment in amazon_sentiments}
    amazon_sentiments.append('positive')
    sentmap.update({'positive': adf[adf['sentiment'] == 'positive'].index.tolist()})
    sdf = criterion_sample(adf, sentmap, amazon_sentiments)
    sdf.to_csv(os.path.join(amazon_processed_path, 'sentiment_sim.tsv'), sep='\t', index=False, header=False)
