import torch


def extract_aspects(dataset, model, device, label_list=["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]):
    data_for_next_dataset = []

    model.eval()
    label_map = {i: label for i, label in enumerate(label_list, 1)}

    for i, inp in enumerate(dataset):
        tokens = dataset.sentences[i]
        num_tokens = len(tokens)
        batch = tuple(t.unsqueeze(0).to(device) if t is not None else None for t in inp)

        with torch.no_grad():
            ate_logits, apc_logits = model(*batch)

        ate_preds = torch.argmax(ate_logits, dim=2)
        ate_preds = ate_preds.detach().cpu().numpy()
        ready_iobs = []
        for j, pred in enumerate(ate_preds[0]):
            if not j:
                continue
            if len(ready_iobs) == num_tokens:
                break
            ready_iobs.append(label_map.get(pred, 'O'))
        data_for_next_dataset.append((tokens, ready_iobs, None))

    return data_for_next_dataset


def classify_polarity(dataset, model, device, sentiments={0: 'Negative', 1: "Neutral", 2: 'Positive', -1: ''}):
    result = []

    model.eval()
    for i, inp in enumerate(dataset):
        batch = tuple(t.unsqueeze(0).to(device) if t is not None else None for t in inp)

        with torch.no_grad():
            ate_logit, apc_logit = model(*batch)

        apc_pred = int(torch.argmax(apc_logit, -1))
        result.append((dataset.sentences[i], dataset.aspect_text[i], sentiments[apc_pred], apc_pred))

    return result


def extract_aspects_batch(dataloader, model, device, label_list=["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]):
    data_for_next_dataset = []

    model.eval()
    label_map = {i: label for i, label in enumerate(label_list, 1)}

    for i, inp in enumerate(dataloader):
        batch = tuple(t.to(device) if t is not None else None for t in inp[:-1])
        with torch.no_grad():
            ate_logits, apc_logits = model(*batch)

        ate_preds = torch.argmax(ate_logits, dim=2)
        ate_preds = ate_preds.detach().cpu().numpy()

        idxs = inp[-1].tolist()
        for k, ate_pred in enumerate(ate_preds):
            ready_iobs = []
            tokens = dataloader.dataset.sentences[idxs[k]]
            num_tokens = len(tokens)
            for j, pred in enumerate(ate_pred):
                if not j:
                    continue
                if len(ready_iobs) == num_tokens:
                    break
                ready_iobs.append(label_map.get(pred, 'O'))
            data_for_next_dataset.append((tokens, ready_iobs, None))

    return data_for_next_dataset


def classify_polarity_batch(dataloader, model, device, sentiments={0: 'Negative', 1: "Neutral", 2: 'Positive', -1: ''}):
    result = []

    model.eval()
    for i, inp in enumerate(dataloader):
        batch = tuple(t.to(device) if t is not None else None for t in inp[:-1])

        with torch.no_grad():
            _, apc_logits = model(*batch)

        idxs = inp[-1].tolist()
        apc_preds = torch.argmax(apc_logits, -1)
        for k, apc_pred in enumerate(apc_preds):
            result.append((dataloader.dataset.sentences[idxs[k]],
                           dataloader.dataset.aspect_text[idxs[k]],
                           sentiments[int(apc_pred)],
                           int(apc_pred)))

    return result
