from typing import *

import numpy as np
import torch
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.nn.util import get_final_encoder_states
from overrides import overrides

@Seq2VecEncoder.register("transformer-encoder")
class TransformerSeq2VecEncoder(Seq2VecEncoder):
    # All the inputs are taken from StackedSelfAttention
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layers,
                 projection_dim,
                 feedforward_hidden_dim,
                 num_attention_heads):
        super(Seq2VecEncoder, self).__init__()
        self.input_dim = input_dim

        self.output_dim = hidden_dim

        self.encoder = StackedSelfAttentionEncoder(input_dim=input_dim,
                                                   hidden_dim=hidden_dim,
                                                   projection_dim=projection_dim,
                                                   feedforward_hidden_dim=feedforward_hidden_dim,
                                                   num_layers=num_layers,
                                                   num_attention_heads=num_attention_heads) 

    @overrides
    def forward(self,
                inputs,
                mask):
        output_encoder = self.encoder(inputs, mask)
        single_vector_output = get_final_encoder_states(output_encoder, mask)
        return single_vector_output
    def get_input_dim(self):
        return self.input_dim
    def get_output_dim(self):
        return self.output_dim

