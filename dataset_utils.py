import torch
from torch.utils.data import Dataset
import transformers
import numpy as np
import spacy
import networkx as nx

SENTIMENT_PADDING = -1


class ATEPCDataset(Dataset):
    def __init__(self,
                 mode,
                 mode_SRD,
                 sentences,
                 sentences_tags,
                 polarity_labels,
                 tokenizer,
                 max_sent_len,
                 SRD,
                 spacy_model=None,
                 return_idx=False):
        assert mode in ['train_test', 'infer_ate', 'infer_apc']
        assert mode_SRD in ['simple', 'syntax_tree']
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        self.mode = mode
        self.mode_SRD = mode_SRD
        self.SRD = SRD

        if mode == 'train_test':
            self.sentences = sentences
            self.sentences_tags = sentences_tags
            nonzero = [np.flatnonzero(np.array(polarity_labels[i]) - SENTIMENT_PADDING)
                       for i in range(len(self.sentences))]
            self.aspect_text = [[self.sentences[i][j]
                                 for j in nonzero[i]] for i in range(len(self.sentences))]
            self.aspect_tags = [[self.sentences_tags[i][j]
                                 for j in nonzero[i]] for i in range(len(self.sentences_tags))]
            sentences_lefts = [self.sentences[i][:nonzero[i][0]] for i in range(len(self.sentences))]
            self.aspect_starts = [len(tokenizer.tokenize(' '.join(sentences_lefts[i]))) + 1
                                  for i in range(len(self.sentences))]
            self.aspect_lens = [len(tokenizer.tokenize(' '.join(self.aspect_text[i])))
                                for i in range(len(self.sentences))]

        elif mode == 'infer_ate':
            self.sentences = sentences
            self.sentences_tags = None
            self.aspect_text = None
            self.aspect_tags = None
            self.aspect_starts = None
            self.aspect_lens = None
        else:
            self.sentences = []
            self.sentences_tags = []
            self.aspect_text = []
            self.aspect_tags = []
            self.aspect_starts = []
            self.aspect_lens = []
            for sent_i, sentence_tag in enumerate(sentences_tags):
                for i in range(len(sentence_tag)):
                    if sentence_tag[i] == 'B-ASP':
                        self.sentences.append(sentences[sent_i])
                        self.aspect_starts.append(len(tokenizer.tokenize(' '.join(sentences[sent_i][:i]))) + 1)
                        aspect_text = [sentences[sent_i][i]]
                        aspect_tag = ['B-ASP']
                        sentence_tag_to_append = ['O'] * i + ['B-ASP']
                        j = i + 1
                        while j < len(sentence_tag) and sentence_tag[j] == 'I-ASP':
                            sentence_tag_to_append.append('I-ASP')
                            aspect_text.append(sentences[sent_i][j])
                            aspect_tag.append('I-ASP')
                            j += 1
                        sentence_tag_to_append += ['O'] * (len(sentence_tag) - len(sentence_tag_to_append))
                        self.sentences_tags.append(sentence_tag_to_append)
                        self.aspect_text.append(aspect_text)
                        self.aspect_tags.append(aspect_tag)
                        self.aspect_lens.append(len(tokenizer.tokenize(' '.join(aspect_text))))

        if mode == 'train_test':
            self.polarity_labels = polarity_labels
        else:
            self.polarity_labels = None

        self.tokenizer = tokenizer
        self.tokenizer.bos_token = '[CLS]'
        self.tokenizer.eos_token = '[SEP]'
        self.max_sent_len = max_sent_len
        self.iob_tags_set = ["O", "B-ASP", "I-ASP", self.tokenizer.bos_token, self.tokenizer.eos_token]
        if spacy_model:
            self.nlp = spacy.load(spacy_model)

        self.return_idx = return_idx

    def __get_cdm_simple(self, input_ids, index):
        aspect_start = self.aspect_starts[index]
        aspect_len = self.aspect_lens[index]
        cdm = np.zeros(self.max_sent_len)
        local_context_start = max(0, aspect_start - self.SRD)
        local_context_end = min(aspect_start + aspect_len + self.SRD - 1, self.max_sent_len)

        text_len = np.count_nonzero(input_ids) - aspect_len - 1

        for i in range(min(text_len, self.max_sent_len)):
            if local_context_start <= i <= local_context_end:
                cdm[i] = 1

        return cdm

    def __get_cdw_simple(self, input_ids, index):
        aspect_start = self.aspect_starts[index]
        aspect_len = self.aspect_lens[index]
        cdw = np.zeros(self.max_sent_len)
        local_context_start = max(0, aspect_start - self.SRD)
        local_context_end = min(aspect_start + aspect_len + self.SRD - 1, self.max_sent_len)

        text_len = np.count_nonzero(input_ids) - aspect_len - 1

        for i in range(min(text_len, self.max_sent_len)):
            if i < local_context_start:
                w = 1 - (local_context_start - i) / text_len
            elif local_context_start <= i <= local_context_end:
                w = 1
            else:
                w = 1 - (i - local_context_end) / text_len
            cdw[i] = w

        return cdw

    def __calculate_syntax_distance(self, index):
        sentence = ' '.join(self.sentences[index])
        aspect_lowered = [a.lower() for a in self.aspect_text[index]]
        doc = self.nlp(sentence)
        edges = []
        cnt = 0
        term_ids = [0] * len(aspect_lowered)
        for token in doc:
            # Record the position of aspect terms
            if cnt < len(aspect_lowered) and token.lower_ == aspect_lowered[cnt]:
                term_ids[cnt] = token.i
                cnt += 1

            for child in token.children:
                edges.append(('{}_{}'.format(token.lower_, token.i),
                              '{}_{}'.format(child.lower_, child.i)))

        graph = nx.Graph(edges)

        dist = [0.0] * len(doc)
        for i, word in enumerate(doc):
            source = '{}_{}'.format(word.lower_, word.i)
            sum = 0
            for term_id, term in zip(term_ids, aspect_lowered):
                target = '{}_{}'.format(term, term_id)
                try:
                    sum += nx.shortest_path_length(graph, source=source, target=target)
                except:
                    sum += len(doc)
            dist[i] = sum / len(aspect_lowered)

        padded_dist = []
        for token, dist_el in zip(doc, dist):
            new_tokens = self.tokenizer.tokenize(token.lower_)
            for _ in range(len(new_tokens)):
                padded_dist.append(dist_el)

        total_dist = [self.max_sent_len] * self.max_sent_len
        padded_dist = padded_dist[:self.max_sent_len - 2]
        padded_dist.insert(0, len(doc))
        padded_dist.append(len(doc))

        total_dist[:len(padded_dist)] = padded_dist
        return total_dist

    def __get_cdm_syntax_tree(self, input_ids, index, dist):
        aspect_len = self.aspect_lens[index]
        cdm = np.zeros(self.max_sent_len)

        text_len = np.count_nonzero(input_ids) - aspect_len - 1

        for i in range(min(text_len, self.max_sent_len)):
            if dist[i] <= self.SRD:
                cdm[i] = 1

        return cdm

    def __get_cdw_syntax_tree(self, input_ids, index, dist):
        aspect_len = self.aspect_lens[index]
        cdw = np.zeros(self.max_sent_len)

        text_len = np.count_nonzero(input_ids) - aspect_len - 1

        for i in range(min(text_len, self.max_sent_len)):
            if dist[i] > self.SRD:
                w = 1 - dist[i] / text_len
                cdw[i] = w
            else:
                cdw[i] = 1

        return cdw

    def __getitem__(self, index):
        iob_tag_map = {label: i for i, label in enumerate(self.iob_tags_set, 1)}
        if self.mode in ['train_test', 'infer_apc']:
            encoding_text = self.sentences[index] + [self.tokenizer.eos_token] + self.aspect_text[index]
            iob_tag_raw = self.sentences_tags[index] + [self.tokenizer.eos_token] + self.aspect_tags[index]
            if len(iob_tag_raw) > self.max_sent_len - 2:
                iob_tag_raw = iob_tag_raw[:self.max_sent_len - 2]
        else:
            encoding_text = self.sentences[index]
            iob_tag_raw = None

        encoding = self.tokenizer(encoding_text,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_sent_len)
        cdw = None
        cdm = None
        if self.mode != 'infer_ate':
            if self.mode_SRD == 'simple':
                cdw = self.__get_cdw_simple(encoding.input_ids, index)
                cdw = torch.tensor(cdw, dtype=torch.float32)

                cdm = self.__get_cdm_simple(encoding.input_ids, index)
                cdm = torch.tensor(cdm, dtype=torch.float32)
            elif self.mode_SRD == 'syntax_tree':
                dist = self.__calculate_syntax_distance(index)

                cdw = self.__get_cdw_syntax_tree(encoding.input_ids, index, dist)
                cdw = torch.tensor(cdw, dtype=torch.float32)

                cdm = self.__get_cdm_syntax_tree(encoding.input_ids, index, dist)
                cdm = torch.tensor(cdm, dtype=torch.float32)

        if iob_tag_raw:
            iob_tag_raw = [self.tokenizer.bos_token] + iob_tag_raw + [self.tokenizer.eos_token]
            iob_tag = [iob_tag_map[label] for label in iob_tag_raw]
            iob_tag += [0] * (self.max_sent_len - len(iob_tag))
        else:
            iob_tag = None

        valid_ids = [1 if encoding.offset_mapping[i][0] == 0 else 0 for i in range(self.max_sent_len)]

        if self.polarity_labels:
            polarity = self.polarity_labels[index][
                np.flatnonzero(np.array(self.polarity_labels[index]) - SENTIMENT_PADDING)[0]]
        else:
            polarity = None

        attention_labels = [1] * (len(encoding_text) + 2)
        if len(attention_labels) > self.max_sent_len:
            attention_labels = attention_labels[:self.max_sent_len]
        attention_labels += [0] * (self.max_sent_len - len(attention_labels))

        input_ids = torch.tensor(encoding.input_ids)
        attention_mask = torch.tensor(encoding.attention_mask)
        token_type_ids = torch.tensor(encoding.token_type_ids)
        if iob_tag:
            iob_tag = torch.tensor(iob_tag)
        else:
            iob_tag = None

        if polarity is not None:
            polarity = torch.tensor(polarity)
        else:
            polarity = None

        valid_ids = torch.tensor(valid_ids)
        attention_labels = torch.tensor(attention_labels)
        if not self.return_idx:
            return input_ids, token_type_ids, attention_mask, iob_tag, polarity, valid_ids, cdm, cdw
        else:
            return input_ids, token_type_ids, attention_mask, iob_tag, polarity, valid_ids, cdm, cdw, index

    def __len__(self):
        return len(self.sentences)
