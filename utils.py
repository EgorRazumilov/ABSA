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

