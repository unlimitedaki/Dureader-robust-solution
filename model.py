from transformers import BertTokenizer,BertModel
import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertOnlyMLMHead
from torch.nn import CrossEntropyLoss
import pdb

class BertForQuestionAnsweringWithMaskedLM(BertPreTrainedModel):
    def __init__(self,config,bert_model = None):
        super(BertForQuestionAnsweringWithMaskedLM,self).__init__(config)
        self.num_labels = config.num_labels
        # self.loss_beta = args.loss_beta
        self.bert = BertModel(config)
        # qa
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # mlm
        self.cls = BertOnlyMLMHead(config)
        # answer content 
        self.answer_content_classifier = nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size,2)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        masked_lm_labels=None,
        answer_content_labels = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2 
            # pdb.set_trace()
            if answer_content_labels is not None:
                logits = self.answer_content_classifier(sequence_output)
                logits = torch.squeeze(logits,-1)
                softmax_logits = F.softmax(logits,dim = -1)
                answer_content_loss = loss_fct(softmax_logits.view(-1,self.num_labels),answer_content_labels.view(-1))
                qa_loss = qa_loss + answer_content_loss
            outputs = (qa_loss,)+ outputs
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            # outputs = (masked_lm_loss,) + outputs
            # total_loss = (masked_lm_loss+qa_loss) /2
            outputs = (masked_lm_loss,) + outputs
        
        return outputs

        
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config, bert_model = None):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
