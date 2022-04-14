from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention
from dataset_utils import SENTIMENT_PADDING
import torch
import torch.nn as nn
import numpy as np
import copy


class SelfAttention(nn.Module):
    def __init__(self, config, max_seq_len, device):
        super(SelfAttention, self).__init__()
        self.torch_device = device
        self.max_seq_len = max_seq_len
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.max_seq_len))
        zero_tensor = torch.tensor(zero_vec).float().to(self.torch_device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_ATEPC(nn.Module):
    def __init__(self, bert_base_model, use_bert_spc, dropout, max_seq_len, LCF, device):
        super(LCF_ATEPC, self).__init__()
        config = bert_base_model.config
        self.bert_for_global_context = bert_base_model
        self.torch_device = device
        assert LCF in ['cdm', 'cdw', 'fusion']
        self.LCF = LCF
        self.use_bert_spc = use_bert_spc
        self.bert_for_local_context = copy.deepcopy(self.bert_for_global_context)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.SA1 = SelfAttention(config, self.max_seq_len, device)
        self.SA2 = SelfAttention(config, self.max_seq_len, device)
        self.hidden_state = config.hidden_size
        self.linear_double = nn.Linear(self.hidden_state * 2, self.hidden_state)
        self.linear_triple = nn.Linear(self.hidden_state * 3, self.hidden_state)

        self.classifier_polarity = torch.nn.Linear(self.hidden_state, 3)
        self.num_labels = 6
        self.classifier_aspect = torch.nn.Linear(self.hidden_state, self.num_labels)

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        # BERT-SPC input to BERT-BASE input
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(self.torch_device)

    def get_ids_for_local_context_extractor(self, text_indices):
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(self.torch_device)

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, polarities=None,
                valid_ids=None, cdm_vec=None, cdw_vec=None, return_all=False):

        cdm_vec = cdm_vec.unsqueeze(2) if cdm_vec is not None else None
        cdw_vec = cdw_vec.unsqueeze(2) if cdw_vec is not None else None
        if not self.use_bert_spc:
            input_ids_spc = self.get_ids_for_local_context_extractor(input_ids_spc)
            labels = self.get_batch_token_labels_bert_base_indices(labels)
        global_context_out = self.bert_for_global_context(input_ids_spc, token_type_ids, attention_mask)['last_hidden_state']

        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.torch_device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier_aspect(global_context_out)

        if cdm_vec is not None or cdw_vec is not None:
            local_context_ids = self.get_ids_for_local_context_extractor(input_ids_spc)
            local_context_out = self.bert_for_local_context(local_context_ids)['last_hidden_state']
            batch_size, max_len, feat_dim = local_context_out.shape
            local_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.torch_device)
            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        local_valid_output[i][jj] = local_context_out[i][j]
            local_context_out = self.dropout(local_valid_output)

            if self.LCF == 'cdm':
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cdm_context_out = self.SA1(cdm_context_out)
                cat_out = torch.cat((global_context_out, cdm_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
            elif self.LCF == 'cdw':
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cdw_context_out = self.SA1(cdw_context_out)
                cat_out = torch.cat((global_context_out, cdw_context_out), dim=-1)
                cat_out = self.linear_double(cat_out)
            elif self.LCF == 'fusion':
                cdm_context_out = torch.mul(local_context_out, cdm_vec)
                cdw_context_out = torch.mul(local_context_out, cdw_vec)
                cat_out = torch.cat((global_context_out, cdw_context_out, cdm_context_out), dim=-1)
                cat_out = self.linear_triple(cat_out)
            sa_out = self.SA2(cat_out)
            pooled_out = self.pooler(sa_out)
            pooled_out = self.dropout(pooled_out)
            apc_logits = self.classifier_polarity(pooled_out)
        else:
            apc_logits = None

        if labels is not None and polarities is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss_sen = nn.CrossEntropyLoss(ignore_index=SENTIMENT_PADDING)
            loss_ate = loss_fct(ate_logits.view(-1, self.num_labels), labels.view(-1))
            loss_apc = loss_sen(apc_logits, polarities)
            if return_all:
                return ate_logits, apc_logits, loss_ate, loss_apc
            else:
                return loss_ate, loss_apc
        else:
            return ate_logits, apc_logits
