# -*- coding: utf-8 -*-
import json
import logging
from overrides import overrides
from typing import Dict, List
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField, ListField, MetadataField, Field
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, Token
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('snli-meta')
class SNLIMetaReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = example["gold_label"]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue

                premise = example["sentence1"]
                hypothesis = example["sentence2"]

                yield self.text_to_instance(premise, hypothesis, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)

        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        if label:
            metadata['label'] = label
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)


@DatasetReader.register('sentence_similarity')
class SentenceSimilarityDatasetReader(DatasetReader):
    """
    Read a txt file containing records in tsv format with the following columns:
    sentence1, sentence2, label
    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence: ``TextField``,
        positive: ``TextField``,
        negatives: ``ListField`` AND
        metadata: ``MetadataField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    lazy : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 num_negative_samples: int = 4,
                 lazy: bool = False,
                 synsample: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._num_negative_samples = num_negative_samples
        self._add_start_token = add_start_token
        self._synsample = synsample

    """
    ``SNLIDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    @overrides
    def _read(self, data_info):
        semantic_path = cached_path(data_info['semantic'])
        syntactic_path = cached_path(data_info['syntactic'])
        logger.info(f"Reading instances from dataset: {data_info}")
        with open(semantic_path) as semp:
            for line in semp:
                semsents = line.split('\t')
                if len(semsents) < 2 + self._num_negative_samples:
                    continue
                yield self.text_to_instance(semsents[0], semsents[1], semsents[2:], True)

        with open(syntactic_path) as synp:
            for line in synp:
                synsents = line.split('\t')
                if len(synsents) < 2 + self._num_negative_samples and self._synsample:
                    continue
                if self._synsample:
                    yield self.text_to_instance(synsents[0], synsents[1], synsents[2:], False)
                else:
                    yield self.text_to_instance(synsents[0], 'positive', ['negatives']*self._num_negative_samples, False)

    @overrides
    def text_to_instance(self, sentence: str, positive: str = None, negatives: List[str] = None, is_semantic: bool = True) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        tokenized_positive = self._tokenizer.tokenize(positive) if positive else None
        if len(tokenized_sentence) > 40:
            tokenized_sentence = tokenized_sentence[:40]
        if tokenized_positive and len(tokenized_positive) > 40:
            tokenized_positive = tokenized_positive[:40]
        tokenized_negatives = [self._tokenizer.tokenize(negative) for negative in negatives] if negatives else None
        if negatives:
            for idx, tokenized_negative in enumerate(tokenized_negatives):
                if len(tokenized_negative) > 40:
                    tokenized_negatives[idx] = tokenized_negative[:40]
        if self._add_start_token:
            tokenized_sentence.insert(0, Token(START_SYMBOL))
            if positive:
                tokenized_positive.insert(0, Token(START_SYMBOL))
            if negatives:
                for tokenized_negative in tokenized_negatives:
                    tokenized_negative.insert(0, Token(START_SYMBOL))
        tokenized_sentence.append(Token(END_SYMBOL))
        if positive:
            tokenized_positive.append(Token(END_SYMBOL))
        if negatives:
            for tokenized_negative in tokenized_negatives:
                tokenized_negative.append(Token(END_SYMBOL))
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        positive_field = TextField(tokenized_positive, self._token_indexers) if positive else None
        negative_fields = [TextField(tokenized_negative,
                                     self._token_indexers) for tokenized_negative in tokenized_negatives] if negatives else None
        negatives_field = ListField(negative_fields) if negatives else None
        metadata_field = MetadataField({'is_semantic': is_semantic})
        return Instance({"sentence": sentence_field, "positive": positive_field,
                         'negatives': negatives_field, 'metadata': metadata_field})


@DatasetReader.register('paraphrase-ref')
class ParaphraseDatasetReader(DatasetReader):
    """
    Read a txt file containing records in tsv format with the following columns:
    sentence1, sentence2, label
    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence: ``TextField``,
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    lazy : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._add_start_token = add_start_token

    """
    ``SNLIDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    @overrides
    def _read(self, data_path):
        data_path = cached_path(data_path)
        logger.info(f"Reading instances from dataset: {data_path}")
        with open(data_path) as df:
            for line in df:
                cols = line.rstrip().split('\t')
                yield self.text_to_instance(cols[0], cols[1], cols[2])

    @overrides
    def text_to_instance(self, sentence: str, synpro: str, semref: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        tokenized_synpro = self._tokenizer.tokenize(synpro)
        tokenized_semref = self._tokenizer.tokenize(semref)
        if self._add_start_token:
            tokenized_sentence.insert(0, Token(START_SYMBOL))
            tokenized_synpro.insert(0, Token(START_SYMBOL))
            tokenized_semref.insert(0, Token(START_SYMBOL))
        tokenized_sentence.append(Token(END_SYMBOL))
        tokenized_synpro.append(Token(END_SYMBOL))
        tokenized_semref.append(Token(END_SYMBOL))
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        synpro_field = TextField(tokenized_synpro, self._token_indexers)
        semref_field = TextField(tokenized_semref, self._token_indexers)
        return Instance({"sentence": sentence_field, "synpro_field": synpro_field, "semref_field": semref_field})


@DatasetReader.register('autoencoder')
class AutoDatasetReader(DatasetReader):
    """
    Read a txt file containing records in tsv format with the following columns:
    sentence1, sentence2, label
    The output of ``read`` is a list of ``Instance`` s with the fields:
        sentence: ``TextField``,
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    lazy : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._add_start_token = add_start_token

    """
    ``SNLIDatasetReader`` class inherits Seq2SeqDatasetReader as the only
    difference is when dealing with autoencoding tasks i.e., the target equals the source.
    """
    @overrides
    def _read(self, data_path):
        data_path = cached_path(data_path)
        logger.info(f"Reading instances from dataset: {data_path}")
        with open(data_path) as df:
            for line in df:
                yield self.text_to_instance(line)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        if self._add_start_token:
            tokenized_sentence.insert(0, Token(START_SYMBOL))
        tokenized_sentence.append(Token(END_SYMBOL))
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        return Instance({"sentence": sentence_field})
