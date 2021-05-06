import torch
import torch.nn as nn
from catalyst.metrics.functional import reciprocal_rank
from catalyst.metrics.functional._misc import process_recsys_components


def evaluate_rec_task_metrics(
    output: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 20,
) -> dict:
    # f1_score
    output_top_k = output.argsort(descending=True)[:, :top_k]
    all_f1 = []
    for p, l in zip(output_top_k, target):
        p = p.cpu().numpy()
        l = l.cpu().numpy()
        nb_hits = len(set(p).intersection(set(l)))
        precision = nb_hits / top_k
        recall = nb_hits / len(set(l)) if len(l) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        all_f1.append(f1)
    f1_score = torch.Tensor(all_f1).mean()

    # mrr
    # TODO: mrr は直前のアイテムしか評価対象にならないので、書き直す必要がある
    target_one_hot = nn.functional.one_hot(target.squeeze_(), num_classes=output.size()[1])
    reciprocal_ranks = reciprocal_rank(output, target_one_hot, k=top_k)
    mrr = reciprocal_ranks.mean()

    metrics = {
        "f1_score": f1_score,
        "mrr": mrr,
    }
    return metrics
