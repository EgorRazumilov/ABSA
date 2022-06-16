import copy
import re

def split_text(text):
    splited = re.split(r'\s*\b\s*', text)
    splited = list(filter(lambda x: len(x) > 0, splited))
    return splited

def is_similar(s1, s2):
    count = 0.0
    for token in s1.split(' '):
        if token in s2:
            count += 1
    if count / len(s1.split(' ')) >= 0.8 and count / len(s2.split(' ')) >= 0.8:
        return True
    else:
        return False



def read_train_test_files(filename):
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag = []
    polarity = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity))
                sentence = []
                tag = []
                polarity = []
            continue
        splits = line.split(' ')
        if len(splits) != 3:
            print('warning! detected error line(s) in input file:{}'.format(line))
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(int(splits[-1][:-1]))

    if len(sentence) > 0:
        data.append((sentence, tag, polarity))
    return data


def read_train_test_files_new(filename):
    f = open(filename, encoding='utf8')
    str2sent = {'-999': -1, 'Negative': 0, 'Neutral': 1, 'Positive': 2}
    data = []
    sentence = []
    tag = []
    polarity = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity))
                sentence = []
                tag = []
                polarity = []
            continue
        splits = line.split(' ')
        if len(splits) != 3:
            print('warning! detected error line(s) in input file:{}'.format(line))
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(str2sent[splits[-1][:-1]])

    if len(sentence) > 0:
        data.append((sentence, tag, polarity))
    return data

def assemble_aspects(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    raw_lines = fin.readlines()
    fin.close()
    i = 0
    lines = []
    while i < len(raw_lines):
        if "$T$" in raw_lines[i]:
            lines.append(raw_lines[i])
            lines.append(raw_lines[i + 1])
            lines.append(raw_lines[i + 2])
            i += 3
        else:
            i += 1
    if len(lines) == 0:
        return None
    for i in range(len(lines)):
        if i % 3 == 0 or i % 3 == 1:
            lines[i] = ' '.join(split_text(lines[i].strip())).replace('$ t $', '$T$').replace('$ T $', '$T$')
        else:
            lines[i] = lines[i].strip()

    def unify_same_samples(same_samples):
        text = same_samples[0][0].replace('$T$', same_samples[0][1])
        polarities = [-999] * len(text.split())
        tags = ['O'] * len(text.split())
        samples = []
        for sample in same_samples:
            # print(sample)
            polarities_tmp = copy.deepcopy(polarities)

            try:
                asp_begin = (sample[0].split().index('$T$'))
                asp_end = sample[0].split().index('$T$') + len(sample[1].split())
                for i in range(asp_begin, asp_end):
                    polarities_tmp[i] = sample[2]
                    if i - sample[0].split().index('$T$') < 1:
                        tags[i] = 'B-ASP'
                    else:
                        tags[i] = 'I-ASP'
                samples.append([text, tags, polarities_tmp])
            except:
                print('Ignore Error:', sample[0], fname)

        return samples

    samples = []
    aspects_in_one_sentence = []
    for i in range(0, len(lines), 3):

        lines[i] = lines[i].replace('$T$', ' $T$ ').replace('  ', ' ')

        if len(aspects_in_one_sentence) == 0:
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
            continue
        if is_similar(aspects_in_one_sentence[-1][0], lines[i]):
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
        else:
            samples.extend(unify_same_samples(aspects_in_one_sentence))
            aspects_in_one_sentence = []
            aspects_in_one_sentence.append([lines[i], lines[i + 1], lines[i + 2]])
    samples.extend(unify_same_samples(aspects_in_one_sentence))

    return samples

def edit_samples(samples):
    edited_samples = []
    for sample in samples:
        edited_sample = copy.deepcopy(sample)
        str2sent = {-999: -1, 'Negative': 0, 'Neutral': 1, 'Positive': 2}
        for i, polarity in enumerate(sample[2]):
            edited_sample[2][i] = str2sent[polarity]
        edited_samples.append(edited_sample)
    return edited_samples