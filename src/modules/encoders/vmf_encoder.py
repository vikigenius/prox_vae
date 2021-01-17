#!/usr/bin/env python
import torch
from typing import Optional
from torch.nn import Linear
from src.modules.distributions import VonMisesFisher, HypersphericalUniform
from typing import Dict, Tuple
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from src.modules.encoders.variational_encoder import VariationalEncoder


@VariationalEncoder.register('vmf')
class VMFEncoder(VariationalEncoder):
    def __init__(self,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 latent_dim: int,
                 concentration: Optional[float] = None) -> None:
        super().__init__(text_field_embedder, encoder, latent_dim)
        self._latent_to_mean = Linear(self._encoder.get_output_dim(), self.latent_dim)
        if concentration:
            self._latent_to_concentration = lambda x: concentration
        else:
            self._latent_to_concentration = Linear(self._encoder.get_output_dim(), self.latent_dim)

    @staticmethod
    def reparametrize(prior: HypersphericalUniform,
                      posterior: VonMisesFisher,
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Creating the latent vector using the reparameterization trick
        """
        return posterior.rsample()

    def forward(self, source_tokens: Dict[str, torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make a forward pass of the encoder, then returning the hidden state.
        """
        final_state = self.encode(source_tokens)
        mean = self._latent_to_mean(final_state)
        concentration = self._latent_to_concentration(final_state)
        prior = HypersphericalUniform(self.latent_dim)
        posterior = VonMisesFisher(mean, concentration)
        return {
            'prior': prior,
            'posterior': posterior,
        }
