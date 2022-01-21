import numpy as np
import torch
import torch.nn as nn

from utils import Config
from transformers import AutoTokenizer, AutoModel

class AutoModelForSequenceClassification_SPV_MIP(nn.Module):
    """MelBERT"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_SPV_MIP, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.hidden_linear_1 = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.hidden_linear_2 = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.hidden_linear_3 = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 3, num_labels)
        self._init_weights(self.hidden_linear_1)
        self._init_weights(self.hidden_linear_2)
        self._init_weights(self.hidden_linear_3)

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
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the second input token indicies
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the second input.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        mwe_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        mwe_output = self.dropout(mwe_output)
        pooled_output = self.dropout(pooled_output)

        mwe_output = mwe_output.sum(1) / (1e-16 + target_mask.sum(1).unsqueeze(-1)) # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        mwe_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        mwe_output_2 = self.dropout(mwe_output_2)
        mwe_output_2 = mwe_output_2.sum(1) / (1e-16 + target_mask_2.sum(1).unsqueeze(-1))

        # Get hidden vectors each from SPV and MIP linear layers
        hidden1 = self.hidden_linear_1(torch.cat([pooled_output, mwe_output], dim=1))
        hidden2 = self.hidden_linear_2(torch.cat([mwe_output_2, mwe_output], dim=1))
        hidden3 = self.hidden_linear_3(torch.cat([mwe_output_2, pooled_output], dim=1))

        logits = self.classifier(self.dropout(torch.cat([hidden1, hidden2, hidden3], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits