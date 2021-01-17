import torch
import random
import logging
import numpy
import allennlp.nn.util as nn_util
from overrides import overrides
from typing import Dict, List, Iterable
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.callbacks import Callback, Events, handle_event
from allennlp.training import CallbackTrainer
from allennlp.data import Vocabulary, Instance
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import BLEU, Average
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from src.modules.encoders import VariationalEncoder
from src.modules.decoders import Decoder
from src.modules.annealer import LossWeight
from src.modules.metrics import CosineSimilarity
from src.modules.distributions.divergence import Divergence, CosineDistance
from src.modules.losses import MarginLoss
from src.utils import tensor_utils


logger = logging.getLogger(__name__)


@Model.register('info_vae')
class InfoVAE(Model):
    """
    This ``InfoVAE`` class is a :class:`Model` which implements a Info VAE

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    task_encoder : ``VariationalEncoder``, required
        The task encoder model of which encodes information relevant to the task
    gen_encoder : ``VariationalEncoder``, required
        The generic encoder model which encodes all non task related information
    decoder : ``Decoder``, required
        The variational decoder model of which to pass the the latent variable
    task_kl_weight : ``LossWeight``, required
        The KL weight on the task Latent Space
    gen_kl_weight : ``LossWeight``, required
        The KL weight on the generic Latent Space
    task_temperature : ``float``, optional (default=1.0)
        The temperature on the task Latent Space
    gen_temperature : ``float``, optional (default=1.0)
        The temperature on the generic Latent Space
    neg_namespace: ``str``, optional (default='tokens')
        The namespace of sampled negative sentences
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 task_encoder: VariationalEncoder,
                 gen_encoder: VariationalEncoder,
                 decoder: Decoder,
                 task_kl_weight: LossWeight,
                 gen_kl_weight: LossWeight,
                 task_temperature: float = 1.0,
                 gen_temperature: float = 1.0,
                 task_divergence: Divergence = CosineDistance(),
                 gen_divergence: Divergence = CosineDistance(),
                 task_margin: MarginLoss = MarginLoss(),
                 gen_margin: MarginLoss = MarginLoss(),
                 task_pen_weight: float = 10.0,
                 gen_pen_weight: float = 10.0,
                 neg_namespace: str = 'tokens',
                 ignore_gen: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        self._task_encoder = task_encoder
        self._gen_encoder = gen_encoder
        self._decoder = decoder

        self._task_latent_dim = task_encoder.latent_dim
        self._gen_latent_dim = gen_encoder.latent_dim

        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token)  # pylint: disable=protected-access
        self._bleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._pbleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._task_kl_metric = Average()
        self._gen_kl_metric = Average()

        self._task_pos_sim = CosineSimilarity()
        self._task_neg_sim = CosineSimilarity()
        self._gen_pos_sim = CosineSimilarity()
        self._gen_neg_sim = CosineSimilarity()
        self._task_p_metric = Average()
        self._gen_p_metric = Average()
        self._task_n_metric = Average()
        self._gen_n_metric = Average()

        self._task_pen = Average()
        self._gen_pen = Average()

        self._task_pen_weight = task_pen_weight
        self._gen_pen_weight = gen_pen_weight

        self._task_kl_weight = task_kl_weight
        self._gen_kl_weight = gen_kl_weight

        self._task_temperature = task_temperature
        self._gen_temperature = gen_temperature

        self._task_divergence = task_divergence
        self._gen_divergence = gen_divergence

        self._task_margin = task_margin
        self._gen_margin = gen_margin

        self._neg_namespace = neg_namespace
        self._ignore_gen = ignore_gen
        initializer(self)

    def _encode(self, sentence: Dict[str, torch.LongTensor],
                encoder: VariationalEncoder, temperature: float):
        encoder_outs = encoder(sentence)
        p_z = encoder_outs['prior']
        q_z = encoder_outs['posterior']
        if self.training:
            z = q_z.rsample()
        else:
            z = encoder.reparametrize(p_z, q_z, temperature)

        kld = kl_divergence(q_z, p_z).sum(dim=1).mean()

        return q_z, z, kld

    def get_task_info(self,
                      sentence: Dict[str, torch.LongTensor],
                      positive: Dict[str, torch.LongTensor] = None,
                      negatives: Dict[str, torch.LongTensor] = None,
                      metadata: List[Dict[str, any]] = None) -> Dict[str, any]:
        stqz, stz, stkld = self._encode(sentence, self._task_encoder, self._task_temperature)
        self._task_kl_metric(stkld)

        if metadata is not None:
            mask = torch.tensor([meta['is_semantic'] for meta in metadata], dtype=torch.bool)
        else:
            mask = torch.zeros(stz.size(0), dtype=torch.bool)
        if positive is None or negatives is None or not mask.any():
            task_penalty = 0.0
        else:
            # Mask the distribution
            stqz = tensor_utils.batch_mask(stqz, mask)
            positive = tensor_utils.batch_mask(positive, mask)

            pqz, _, _ = self._encode(positive, self._task_encoder, self._task_temperature)

            pscore = self._task_divergence(stqz, pqz)
            self._task_p_metric(pscore)
            self._task_pos_sim(stqz.mean, pqz.mean)

            # flatten and the negatives
            import pdb
            pdb.set_trace()
            negatives = tensor_utils.batch_mask(negatives, mask)
            sample_size = negatives[self._neg_namespace].size(1)
            negatives = {self._neg_namespace: torch.flatten(negatives[self._neg_namespace], 0, 1)}
            nqz, _, _ = self._encode(negatives, self._task_encoder, self._task_temperature)

            # Expand the sentences
            stqz = Normal(stqz.mean.repeat(sample_size, 1), stqz.stddev.repeat(sample_size, 1))
            nscore = self._task_divergence(stqz, nqz)
            self._task_n_metric(nscore)
            self._task_neg_sim(stqz.mean, nqz.mean)

            task_penalty = self._task_margin(nscore, pscore)
            self._task_pen(task_penalty)

        return stz, self._task_kl_weight.get()*stkld + self._task_pen_weight*task_penalty

    def get_gen_info(self,
                     sentence: Dict[str, torch.LongTensor],
                     positive: Dict[str, torch.LongTensor] = None,
                     negatives: Dict[str, torch.LongTensor] = None,
                     metadata: Dict[str, any] = None) -> Dict[str, any]:
        sgqz, sgz, sgkld = self._encode(sentence, self._gen_encoder, self._gen_temperature)
        self._gen_kl_metric(sgkld)

        if metadata is not None:
            mask = ~torch.tensor([meta['is_semantic'] for meta in metadata], dtype=torch.bool)
        else:
            mask = torch.zeros(sgz.size(0), dtype=torch.bool)
        if positive is None or negatives is None or not mask.any() or self._ignore_gen:
            gen_penalty = 0.0
        else:
            # Mask the distribution
            sgqz = tensor_utils.batch_mask(sgqz, mask)

            positive = tensor_utils.batch_mask(positive, mask)

            pqz, _, _ = self._encode(positive, self._gen_encoder, self._gen_temperature)

            pscore = self._gen_divergence(sgqz, pqz)
            self._gen_p_metric(pscore)
            self._gen_pos_sim(sgqz.mean, pqz.mean)

            # flatten and the negatives

            negatives = tensor_utils.batch_mask(negatives, mask)
            sample_size = negatives[self._neg_namespace].size(1)
            negatives = {self._neg_namespace: torch.flatten(negatives[self._neg_namespace], 0, 1)}
            nqz, _, _ = self._encode(negatives, self._gen_encoder, self._gen_temperature)

            # Expand the sentences
            sgqz = Normal(sgqz.mean.repeat(sample_size, 1), sgqz.stddev.repeat(sample_size, 1))
            nscore = self._gen_divergence(sgqz, nqz)
            self._gen_n_metric(nscore)
            self._gen_neg_sim(sgqz.mean, nqz.mean)

            gen_penalty = self._gen_margin(nscore, pscore)
            self._gen_pen(gen_penalty)
        return sgz, self._gen_kl_weight.get()*sgkld + self._gen_pen_weight*gen_penalty

    def get_paraphrase_info(self, sentence: Dict[str, torch.LongTensor],
                            positive: Dict[str, torch.LongTensor],
                            metadata: Dict[str, any]):
        mask = torch.tensor([meta['is_semantic'] for meta in metadata], dtype=torch.bool)
        _, stz, _ = self._encode(sentence, self._task_encoder, self._task_temperature)
        _, sgpz, _ = self._encode(positive, self._gen_encoder, self._gen_temperature)
        stz = tensor_utils.batch_mask(stz, mask)
        sgpz = tensor_utils.batch_mask(sgpz, mask)
        z = torch.cat([stz, sgpz], dim=1)
        dd = self._decoder(z)
        self._pbleuo(dd['predictions'], tensor_utils.batch_mask(sentence['tokens'], mask))
        self._pbleur(dd['predictions'], tensor_utils.batch_mask(positive['tokens'], mask))
        return z

    def forward(self,
                sentence: Dict[str, torch.LongTensor],
                positive: Dict[str, torch.LongTensor] = None,
                negatives: Dict[str, torch.LongTensor] = None,
                metadata: Dict[str, any] = None,
                synpro: Dict[str, any] = None,
                semref: Dict[str, any] = None, **kwargs) -> Dict[str, any]:
        tz, tloss = self.get_task_info(sentence, positive, negatives, metadata)
        gz, gloss = self.get_gen_info(sentence, positive, negatives, metadata)

        del positive, negatives  # Free up memory

        # Reconstructions
        z = torch.cat([tz, gz], dim=1)
        dd = self._decoder(z, sentence)

        # Delete Unncessary Logits to free up Memory
        del dd['logits']

        loss = (
            dd['loss'] + tloss + gloss
        )
        output_dict = {}
        if self.training:
            self._task_kl_weight.step()
            self._gen_kl_weight.step()
        else:
            self._bleu(dd['predictions'], sentence['tokens'])
            if synpro is not None and semref is not None:
                zsyn, _ = self.get_gen_info(synpro)
                pz = torch.cat([tz, zsyn], dim=1)
                ppreds = self._decoder(pz)['predictions']
                output_dict.update({'paraphrase_predictions': ppreds})
                self._pbleu(ppreds, semref['tokens'])
        output_dict.update({
            'loss': loss,
            'tz': tz,
            'gz': gz,
            'predictions': dd['predictions']
        })
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
            all_metrics.update({'PBLEU': self._pbleu.get_metric(reset=reset)['BLEU']})
        if self.training:
            all_metrics.update({'_tklw': float(self._task_kl_weight.get())})
            all_metrics.update({'_gklw': float(self._gen_kl_weight.get())})
            all_metrics.update({'_tpsim': float(self._task_pos_sim.get_metric(reset=reset))})
            all_metrics.update({'_tnsim': float(self._task_neg_sim.get_metric(reset=reset))})
            all_metrics.update({'_gpsim': float(self._gen_pos_sim.get_metric(reset=reset))})
            all_metrics.update({'_gnsim': float(self._gen_neg_sim.get_metric(reset=reset))})
            all_metrics.update({'_tpscore': float(self._task_p_metric.get_metric(reset=reset))})
            all_metrics.update({'_tnscore': float(self._task_n_metric.get_metric(reset=reset))})
            all_metrics.update({'_gpscore': float(self._gen_p_metric.get_metric(reset=reset))})
            all_metrics.update({'_gnscore': float(self._gen_n_metric.get_metric(reset=reset))})
            all_metrics.update({'tpen': float(self._task_pen.get_metric(reset=reset))})
            all_metrics.update({'gpen': float(self._gen_pen.get_metric(reset=reset))})
        all_metrics.update({'tkl': float(self._task_kl_metric.get_metric(reset=reset))})
        all_metrics.update({'gkl': float(self._gen_kl_metric.get_metric(reset=reset))})
        return all_metrics

    def get_samples(self, latent=None, num_to_sample: int = 1):
        cuda_device = self._get_prediction_device()
        if latent is None:
            prior = Normal(torch.zeros((num_to_sample, self._task_latent_dim + self._gen_latent_dim)),
                           torch.ones((num_to_sample, self._task_latent_dim + self._gen_latent_dim)))
            latent = prior.sample()

        generated = self._decoder.generate(nn_util.move_to_device(latent, cuda_device))

        sentence = self.decode(generated)['predicted_sentences']
        return sentence

    def get_paraphrase_samples(self, task_latent=None, num_to_sample: int = 1):
        cuda_device = self._get_prediction_device()
        if task_latent is None:
            task_prior = Normal(torch.zeros((num_to_sample, self._task_latent_dim)),
                                torch.ones((num_to_sample, self._task_latent_dim)))
            task_latent = task_prior.sample()

        gen_prior = Normal(torch.zeros((num_to_sample, self._gen_latent_dim)),
                           torch.ones((num_to_sample, self._gen_latent_dim)))

        if task_latent.dim() == 1:
            task_latent = task_latent.unsqueeze(0)
        latent1 = torch.cat([task_latent, gen_prior.sample()], dim=1)
        latent2 = torch.cat([task_latent, gen_prior.sample()], dim=1)
        generated1 = self._decoder.generate(nn_util.move_to_device(latent1, cuda_device))
        generated2 = self._decoder.generate(nn_util.move_to_device(latent2, cuda_device))

        sentence1 = self.decode(generated1)['predicted_sentences']
        sentence2 = self.decode(generated2)['predicted_sentences']
        return sentence1, sentence2

    def decode_predictions(self, predictions):
        predicted_indices = predictions
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_sentences = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]

            # Collect indices after the first end_symbol
            if self._start_index in indices:
                indices = indices[indices.index(self._start_index) + 1:]

            predicted_tokens = [self.vocab.get_token_from_index(x) for x in indices]
            all_predicted_sentences.append(' '.join(predicted_tokens))
        return all_predicted_sentences

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        output_dict["predicted_sentences"] = self.decode_predictions(output_dict['predictions'])
        if 'paraphrase_predictions' in output_dict:
            output_dict["predicted_paraphrases"] = self.decode_predictions(output_dict['paraphrase_predictions'])
        return output_dict


@Callback.register("generate_paraphrases")
class ParaphraseGen(Callback):
    """
    This callback handles generating of sample paraphrases
    """
    def __init__(self,
                 num_samples: int = 1):
        self.num_samples = num_samples

    @handle_event(Events.VALIDATE, priority=1000)
    def generate_paraphrase(self, trainer: 'CallbackTrainer'):
        logger.info("generating sample paraphrase")
        trainer.model.eval()
        s1, s2 = trainer.model.get_paraphrase_samples(num_to_sample=self.num_samples)
        logger.info(f'{s1} <------> {s2}')


@Callback.register("generate_conditional_paraphrase")
class ConditionalParaphraseGen(Callback):
    """
    This Callbacks handles generation of conditional paraphrases
    """
    def __init__(self,
                 validation_data: Iterable[Instance]):
        self.instances = validation_data

    @handle_event(Events.VALIDATE, priority=1000)
    def generate_sample(self, trainer: 'CallbackTrainer'):
        logger.info("generating conditional paraphrase")
        trainer.model.eval()
        sample_instances = random.sample(self.instances, 1)
        paraphrase = trainer.model.forward_on_instances(sample_instances)[0]['predicted_paraphrases']
        sentence_tokens = [str(token) for token in sample_instances[0]['sentence']]
        ref_tokens = [str(token) for token in sample_instances[0]['semref']]
        reference = ' '.join(ref_tokens[1:-1])
        syn_tokens = [str(token) for token in sample_instances[0]['synpro']]
        syntax = ' '.join(syn_tokens[1:-1])
        sentence = ' '.join(sentence_tokens[1:-1])
        logger.info(f'{sentence} | {syntax} <-> {paraphrase} | {reference}')


@Callback.register("generate_sample_reconstruction")
class SampleReconstruct(Callback):
    """
    This callback handles generating of sample reconstructions
    """
    def __init__(self,
                 validation_data: Iterable[Instance],
                 num_samples: int = 1):
        self.num_samples = num_samples
        self.instances = validation_data

    def _display_reconstructions(self, instances: Instance,
                                 output_dict: Dict[str, torch.Tensor]):
        for idx, instance in enumerate(instances):
            sentence_tokens = [str(token) for token in instance['sentence']]
            reconstruction = output_dict[idx]['predicted_sentences']
            sentence = ' '.join(sentence_tokens[1:-1])
            logger.info(f'{sentence} <-> {reconstruction}')

    @handle_event(Events.VALIDATE, priority=1000)
    def generate_sample(self, trainer: 'CallbackTrainer'):
        logger.info("generating sample reconstruction")
        trainer.model.eval()
        sample_instances = random.sample(self.instances, self.num_samples)
        output_dicts = trainer.model.forward_on_instances(sample_instances)
        self._display_reconstructions(sample_instances, output_dicts)
