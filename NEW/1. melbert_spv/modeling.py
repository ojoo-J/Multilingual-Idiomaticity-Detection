import numpy as np
import torch
import torch.nn as nn

from utils import Config
from transformers import AutoTokenizer, AutoModel

class AutoModelForSequenceClassification_SPV(nn.Module):
    """MelBERT with only SPV"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_SPV, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        target_mask,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target wor. 1 for target word and 0 otherwise.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1].
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        mwe_output = sequence_output * target_mask.unsqueeze(2)  # [batch, hidden]

        # dropout
        mwe_output = self.dropout(mwe_output)
        pooled_output = self.dropout(pooled_output)

        # Get mean value of target output if the target output consistst of more than one token
        #target_output = target_output.mean(1)
        mwe_output = mwe_output.sum(1) / (1e-16 + target_mask.sum(1).unsqueeze(-1))

        logits = self.classifier(torch.cat([pooled_output, mwe_output], dim=1))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
