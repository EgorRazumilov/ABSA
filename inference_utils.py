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


def classify_polarity(dataset, model, device):
    result = []

    model.eval()
    sentiments = {0: 'Negative', 1: "Neutral", 2: 'Positive', -1: ''}
    for i, inp in enumerate(dataset):
        batch = tuple(t.unsqueeze(0).to(device) if t is not None else None for t in inp)

        with torch.no_grad():
            ate_logit, apc_logit = model(*batch)

        apc_pred = int(torch.argmax(apc_logit, -1))
        result.append((dataset.sentences[i], dataset.aspect_text[i], sentiments[apc_pred], apc_pred))

    return result