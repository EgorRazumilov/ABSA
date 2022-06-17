import torch
from sklearn.metrics import f1_score
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm
from seqeval.metrics import f1_score as f1_sec

def train_loop(model, dataloader, optimizer, device='cuda'):
    model.train()
    ate_loss_av_tr = 0
    apc_loss_av_tr = 0

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        loss_ate, loss_apc = model(*batch)
        loss = loss_ate + loss_apc

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ate_loss_av_tr += loss_ate.detach().cpu()
        apc_loss_av_tr += loss_apc.detach().cpu()

    ate_loss_av_tr /= len(dataloader)
    apc_loss_av_tr /= len(dataloader)
    return ate_loss_av_tr, apc_loss_av_tr


def eval_loop(model, dataloader, eval_ATE=False, eval_ASC=True, device='cuda',
              LABEL_LIST = ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]):

    label_map = {i: label for i, label in enumerate(LABEL_LIST, 1)}
    all_preds_ASC = []
    all_trues_ASC = []
    all_preds_ATE = []
    all_trues_ATE = []

    f1_ATE = None
    f1_ASC = None

    ate_loss_av_val = 0
    apc_loss_av_val = 0
    model.eval()

    for step, batch in enumerate(dataloader):
        iob_ids = batch[3]
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            ate_logits, apc_logits, ate_loss, apc_loss = model(*batch, return_all=True)
            ate_loss_av_val += ate_loss.detach().cpu()
            apc_loss_av_val += apc_loss.detach().cpu()
        if eval_ATE:
            label_ids = model.get_batch_token_labels_bert_base_indices(iob_ids)
            label_ids = label_ids.to('cpu').numpy()
            ate_logits = torch.argmax(ate_logits, dim=2)
            ate_logits = ate_logits.detach().cpu().numpy()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == 5:
                        all_trues_ATE.append(temp_1)
                        all_preds_ATE.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map.get(label_ids[i][j], 'O'))
                        temp_2.append(label_map.get(ate_logits[i][j], 'O'))
        if eval_ASC:
            all_preds_ASC.extend(torch.argmax(apc_logits, -1).tolist())
            all_trues_ASC.extend(batch[4].tolist())

    if eval_ATE:
        f1_ATE = f1_sec(all_trues_ATE, all_preds_ATE)
    if eval_ASC:
        f1_ASC = f1_score(all_trues_ASC, all_preds_ASC, average='macro')
    ate_loss_av_val /= len(dataloader)
    apc_loss_av_val /= len(dataloader)

    return ate_loss_av_val, apc_loss_av_val, f1_ATE, f1_ASC


def train_eval(model,
               dataloader_train,
               dataloader_val,
               optimizer,
               n_epochs,
               path_to_save,
               scheduler=None,
               device='cuda',
               eval_ATE=False,
               eval_ASC=True,
               eval_strategy='best_ASC'):

    assert eval_strategy in ['best_ASC', 'best_ATE', 'best_sum']

    ATE_losses_train = []
    ASC_losses_train = []

    ATE_losses_val = []
    ASC_losses_val = []

    f1_ATE_history = []
    f1_ASC_history = []

    best_metric = 1e9
    for _ in tqdm(range(n_epochs)):
        ate_loss_av_tr, apc_loss_av_tr = train_loop(model, dataloader_train, optimizer, device=device)
        ATE_losses_train.append(ate_loss_av_tr)
        ASC_losses_train.append(apc_loss_av_tr)

        if scheduler is not None:
            scheduler.step()

        ate_loss_av_val, apc_loss_av_val, f1_ATE, f1_ASC = eval_loop(model,
                                                                     dataloader_val,
                                                                     eval_ATE=eval_ATE,
                                                                     eval_ASC=eval_ASC,
                                                                     device=device)
        ATE_losses_val.append(ate_loss_av_val)
        ASC_losses_val.append(apc_loss_av_val)
        f1_ATE_history.append(f1_ATE)
        f1_ASC_history.append(f1_ASC)

        if eval_strategy == 'best_ATE':
            if ate_loss_av_val < best_metric:
                best_metric = ate_loss_av_val
                torch.save(model.state_dict(), path_to_save)

        elif eval_strategy == 'best_ASC':
            if apc_loss_av_val < best_metric:
                best_metric = apc_loss_av_val
                torch.save(model.state_dict(), path_to_save)

        elif eval_strategy == 'best_sum':
            if ate_loss_av_val + apc_loss_av_val < best_metric:
                best_metric = ate_loss_av_val + apc_loss_av_val
                torch.save(model.state_dict(), path_to_save)

        clear_output(True)
        plt.figure(figsize=(10, 5))
        plt.title("ATE and APC Losses")
        plt.plot(ATE_losses_train, label="ATE_train")
        plt.plot(ASC_losses_train, label="ASC_train")

        plt.plot(ATE_losses_val, label="ATE_val")
        plt.plot(ASC_losses_val, label="ASC_val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.title("Metrics")
        if eval_ATE:
            plt.plot(f1_ATE_history, label="ATE_F1")
        if eval_ASC:
            plt.plot(f1_ASC_history, label="ASC_f1")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.show()



