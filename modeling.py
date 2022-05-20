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
        #self.static_embedding = self.encoder.embeddings.word_embeddings
        #self.static_embedding.weight.requires_grad = False
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.hidden_linear_1 = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.hidden_linear_2 = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        #self.hidden_linear_3 = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        #self.relu = nn.ReLU()
        self.layer = nn.Linear(args.classifier_hidden * 4, args.classifier_hidden * 3)
        self.classifier = nn.Linear(args.classifier_hidden * 3, num_labels)
        self._init_weights(self.hidden_linear_1)
        self._init_weights(self.hidden_linear_2)
        self._init_weights(self.layer)
        self._init_weights(self.classifier)
        #self._init_weights(self.hidden_linear_3)


        self.logsoftmax = nn.LogSoftmax(dim=1)
        

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
        input_ids_3,
        input_ids_4,
        target_mask,
        target_mask_2,
        target_mask_3,
        target_mask_4,
        attention_mask,
        attention_mask_2,
        attention_mask_3,
        attention_mask_4, 
        token_type_ids,
        token_type_ids_2,
        labels=None,
        head_mask=None,
    ):

        #### First encoder for full sentence
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


        #### Second encoder 
        outputs_2 = self.encoder(input_ids_2, token_type_ids=token_type_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]

        pooled_output_2 = outputs_2[1]  # [batch, hidden]
        pooled_output_2 = self.dropout(pooled_output_2)

        mwe_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        mwe_output_2 = self.dropout(mwe_output_2)
        mwe_output_2 = mwe_output_2.sum(1) / (1e-16 + target_mask_2.sum(1).unsqueeze(-1)) # [batch, hidden]


        #### Third encoder
        outputs_3 = self.encoder(input_ids_3, attention_mask=attention_mask_3, head_mask=head_mask)
        
        pooled_output_3 = outputs_3[1]  # [batch, hidden]
        pooled_output_3 = self.dropout(pooled_output_3)


        #### Fourth encoder
        outputs_4 = self.encoder(input_ids_4, attention_mask=attention_mask_4, head_mask=head_mask)
        sequence_output_4 = outputs_4[0]  # [batch, max_len, hidden]
        mwe_output_4 = sequence_output_4 * target_mask_4.unsqueeze(2)
        mwe_output_4 = self.dropout(mwe_output_4)
        mwe_output_4 = mwe_output_4.sum(1) / (1e-16 + target_mask_4.sum(1).unsqueeze(-1))


        #static_target_output = self.static_embedding(input_ids_4) * target_mask_4.unsqueeze(2)
        #static_target_output = self.dropout(static_target_output)
        #static_target_output = static_target_output.sum(1) / (1e-16 + target_mask_4.sum(1).unsqueeze(-1))
        
        # Get hidden vectors each from SPV and MIP linear layers
        hidden1 = self.hidden_linear_1(torch.cat([pooled_output, mwe_output], dim=1))
        hidden2 = self.hidden_linear_2(torch.cat([pooled_output_2, mwe_output_2], dim=1))
        #hidden3 = self.hidden_linear_3(torch.cat([pooled_output_3, mwe_output_4], dim=1))

        #hidden1 = self.relu(hidden1)
        #hidden2 = self.relu(hidden2)

        layer = self.layer(self.dropout(torch.cat([hidden1, hidden2, pooled_output_3, mwe_output_4], dim=1)))
        logits = self.classifier(layer)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits