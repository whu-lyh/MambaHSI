
from utils.lovasz_loss import lovasz_softmax
from utils.wrappers import resize


def head_loss(loss_func, logits, label, align_corners=True, require_resize=True):
    if require_resize: # basically used for FCN networks
        seg_logits = resize(
            input=logits,
            size=label.shape[1:],
            mode='bilinear',
            align_corners=align_corners)
    else:
        seg_logits = logits

    loss = loss_func(seg_logits, label)
    if False: # useless till now 20250310
        lloss = lovasz_softmax(seg_logits, label, ignore=255)
        loss = loss + 0.75 * lloss
    return loss