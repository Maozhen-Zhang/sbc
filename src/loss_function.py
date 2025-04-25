from torch.nn import functional as F

def cosine_triplet_loss(anchor, positive, negative, margin=0.1):
    """
    Compute the cosine triplet loss given batches of anchor, positive, and negative samples.

    Args:
    anchor, positive, negative: torch.Tensor, all of shape (batch_size, feature_dim)
    margin: float, margin by which positives should be closer to the anchors than negatives

    Returns:
    torch.Tensor, scalar tensor containing the loss
    """
    # 计算锚点和正例之间的余弦相似度
    pos_similarity = F.cosine_similarity(anchor, positive)
    # 计算锚点和反例之间的余弦相似度
    neg_similarity = F.cosine_similarity(anchor, negative)

    # 计算三元组损失
    losses = F.relu(neg_similarity - pos_similarity + margin)

    # 取均值作为最终的损失
    return losses.mean()


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute the triplet loss as defined by the formula:
    L = max(d(a, p) - d(a, n) + margin, 0)
    where 'd' is the Euclidean distance, 'a' is the anchor, 'p' is the positive sample,
    'n' is the negative sample, and 'margin' is the margin parameter.
    """
    distance_positive = (anchor - positive).pow(2).sum(1)  # Squared Euclidean Distance between anchor and positive
    distance_negative = (anchor - negative).pow(2).sum(1)  # Squared Euclidean Distance between anchor and negative

    losses = F.relu(distance_positive - distance_negative + margin)

    return losses.mean()


def kl_divergence_loss(logits, logits2):
    # 将logits和logits2转换为概率分布
    p = F.softmax(logits, dim=-1)  # softmax to get probabilities for logits
    q = F.softmax(logits2, dim=-1)  # softmax to get probabilities for logits2

    # 使用KL散度计算损失（注意kl_div是一个对数概率损失，所以我们使用log_softmax和softmax计算）
    loss = F.kl_div(F.log_softmax(logits, dim=-1), q, reduction='batchmean')
    return loss