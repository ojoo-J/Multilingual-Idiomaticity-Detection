import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Config
from transformers import AutoTokenizer, AutoModel


class ElemwiseBilinearMatching(nn.Module):
    def __init__(self, input_dim, m, dropout_prob):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = m * input_dim
        self.m = m
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.weight = nn.Parameter(torch.FloatTensor(m * input_dim, 3, 3))
        self.reset_parameters()

    def reset_parameters(self):
        import math
        nn.init.normal(self.weight.data, mean=0, std=math.sqrt(2.0 / 3.0))

    def forward(self, s1, s2):
        batch_size = s1.size(0)
        s1m = s1.repeat(1, self.m)
        s2m = s2.repeat(1, self.m)
        ones = torch.ones_like(s1m)
        # f: (batch_size, m * input_dim, 3)
        f = torch.stack([s1m, s2m, ones], dim=2)
        # f_flat: (batch_size * m * input_dim, 1, 3)
        f_flat = f.view(-1, 1, 3)
        # weight_expand: (batch_size, m * input_dim, 3, 3)
        weight_expand = self.weight.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # weight_expand_flat: (batch_size * m * input_dim, 3, 3)
        weight_expand_flat = weight_expand.view(-1, 3, 3)
        # z: (batch_size, m * input_dim, 3)
        f_flat = self.dropout(f_flat)
        z = f_flat.bmm(weight_expand_flat).view(batch_size, -1, 3)
        # o: (batch_size, m * input_dim)
        z = self.dropout(z)
        # o = (z * self.dropout(f)).sum(2).tanh()
        o = (z * self.dropout(f)).sum(2).tanh()
        return o


class AutoModelForSequenceClassification(nn.Module):
    """Base model for sequence classification"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
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
            target_mask=None,
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
                It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch.
                It's the mask that we typically use for attention when a batch has varying length sentences.
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
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForTokenClassification(nn.Module):
    """Base model for token classification"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
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
                It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch.
                It's the mask that we typically use for attention when a batch has varying length sentences.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)
        target_output = target_output.sum(1) / target_mask.sum()  # [batch, hideen]

        logits = self.classifier(target_output)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_SPV(nn.Module):
    """MelBERT with only SPV"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_SPV, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.static_embedding = self.encoder.embeddings.word_embeddings
        self.static_embedding.weight.requires_grad = False
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size * 4, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)
        # self.elbis = ElemwiseBilinearMatching(config.hidden_size, 3, 0.1)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, target_mask, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
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
            output_hidden_states=True
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        # pooled_output = outputs[1]  # [batch, hidden]

        input_mask = torch.where(input_ids == 1, 0, 1).unsqueeze(-1)
        pooled_output = (sequence_output * input_mask).sum(1) / (1e-16 + input_mask.sum(1))
        # pooled_output = (sequence_output * input_mask).max(1)[0]

        # Get target output with target mask
        sentence_mask = torch.where(token_type_ids > 0, 1, 0).unsqueeze(-1)
        sentence_output = (sequence_output * sentence_mask).sum(1) / (1e-16 + sentence_mask.sum(1))
        target_output = sequence_output * target_mask.unsqueeze(2)  # [batch, hidden]
        # static_target_output = outputs['hidden_states'][0] * target_mask.unsqueeze(2)
        static_target_output = self.static_embedding(input_ids) * target_mask.unsqueeze(2)

        # Get mean value of target output if the target output consist of more than one token
        # target_output = target_output.mean(1)
        target_output = target_output.sum(1) / (1e-16 + target_mask.sum(1).unsqueeze(-1))
        static_target_output = static_target_output.sum(1) / (1e-16 + target_mask.sum(1).unsqueeze(-1))

        # logits = self.classifier(self.elbis(target_output, pooled_output))
        logits = self.classifier(
            self.dropout(torch.cat([pooled_output, target_output, static_target_output, sentence_output], dim=1)))
        # logits = self.classifier(self.dropout(torch.cat([pooled_output, target_output, static_target_output, sentence_output], dim=1)))
        # logits = self.classifier(torch.cat([target_output, pooled_output, target_output * pooled_output, torch.abs(target_output - pooled_output)], dim=1))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_MIP(nn.Module):
    """MelBERT with only MIP"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_MIP, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args
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

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)
        target_output = target_output.sum(1) / target_mask.sum()  # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.sum(1) / target_mask_2.sum()

        logits = self.classifier(torch.cat([target_output_2, target_output], dim=1))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


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

        self.SPV_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.MIP_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

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

        # Get target output with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        target_output = target_output.mean(1)  # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.mean(1)

        # Get hidden vectors each from SPV and MIP linear layers
        SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
