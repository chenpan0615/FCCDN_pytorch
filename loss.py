import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_true, y_pred):
        return self.soft_dice_loss(y_true, y_pred.to(dtype=torch.float32))


class MultiClass_DiceLoss(nn.Module):
    def __init__(self, 
                weight: torch.Tensor, 
                batch: Optional[bool] = True, 
                ignore_index: Optional[int] = -1,
                do_sigmoid: Optional[bool] = False,
                **kwargs,
                )->torch.Tensor:
        super(MultiClass_DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.do_sigmoid = do_sigmoid
        self.binary_diceloss = dice_loss(batch)

    def __call__(self, y_pred, y_true):
        if self.do_sigmoid:
            y_pred = torch.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true.long(), y_pred.shape[1]).permute(0,3,1,2)
        total_loss = 0.0
        tmp_i = 0.0
        for i in range(y_pred.shape[1]):
            if i != self.ignore_index:
                diceloss = self.binary_diceloss(y_pred[:, i, :, :], y_true[:, i, :, :])
                total_loss += torch.mul(diceloss, self.weight[i])
                tmp_i += 1.0
        return total_loss / tmp_i


class dice_bce_loss(nn.Module):
    """Binary"""
    def __init__(self):
        super(dice_bce_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels, do_sigmoid=True):

        if len(scores.shape) > 3:
            scores = scores.squeeze(1)
        if len(labels.shape) > 3:
            labels = labels.squeeze(1)
        if do_sigmoid:
            scores = torch.sigmoid(scores.clone())
        diceloss = self.binnary_dice(scores, labels)
        bceloss = self.bce_loss(scores, labels)
        return diceloss + bceloss


class mc_dice_bce_loss(nn.Module):
    """multi-class"""
    def __init__(self, weight, do_sigmoid = True):
        super(mc_dice_bce_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight)
        self.dice = MultiClass_DiceLoss(weight, do_sigmoid)

    def __call__(self, scores, labels):

        if len(scores.shape) < 4:
            scores = scores.unsqueeze(1)
        if len(labels.shape) < 4:
            labels = labels.unsqueeze(1)
        diceloss = self.dice(scores, labels)
        bceloss = self.ce_loss(scores, labels)
        return diceloss + bceloss


def FCCDN_loss_BCD(scores, labels):
    # scores = model(input)
    # labels = [binary_cd_labels, binary_cd_labels_downsampled2times]
    """ for binary change detection task"""
    criterion = dice_bce_loss()
    loss_change = torch.tensor(0.0).cuda()
    loss_aux = torch.tensor(0.0).cuda()
    # change loss
    loss_change = criterion(scores[0], labels[0])
    # seg map
    out1 = torch.sigmoid(scores[1]).clone()
    out2 = torch.sigmoid(scores[2]).clone()
    out3 = out1.clone()
    out4 = out2.clone()
    
    out1[labels[1]==1]=0
    out2[labels[1]==1]=0
    out3[labels[1]!=1]=0
    out4[labels[1]!=1]=0

    pred_seg_pre_tmp1 = torch.ones(out1.shape).cuda()
    pred_seg_pre_tmp1[out1<=0.5]=0
    pred_seg_post_tmp1 = torch.ones(out2.shape).cuda()
    pred_seg_post_tmp1[out2<=0.5]=0
    
    pred_seg_pre_tmp2 = torch.ones(scores[1].shape).cuda()
    pred_seg_pre_tmp2[out3<=0.5]=0
    pred_seg_post_tmp2 = torch.ones(scores[2].shape).cuda()
    pred_seg_post_tmp2[out4<=0.5]=0

    # seg loss
    loss_aux = 0.2*criterion(out1, pred_seg_post_tmp1, False)
    loss_aux += 0.2*criterion(out2, pred_seg_pre_tmp1, False)
    loss_aux += 0.2*criterion(out3, labels[1]-pred_seg_post_tmp2, False)
    loss_aux += 0.2*criterion(out4, labels[1]-pred_seg_pre_tmp2, False)

    loss = loss_change + loss_aux
    return loss


def FCCDN_loss_SCD(scores, labels):
    # scores = model(input)
    # labels = [binary_cd_labels, segmentation_labels_of_pretemporal, segmentation_labels_of_posttemporal]
    """ for semantic change detection task"""
    criterion = mc_dice_bce_loss()

    pred_seg_pre_unchange = torch.argmax(scores[1], axis=1)
    pred_seg_post_unchange = torch.argmax(scores[2], axis=1)

    pred_seg_pre_unchange[labels[0][:,0,: :]==1] = 0
    pred_seg_post_unchange[labels[0][:,0,: :]==1] = 0

    aux_loss = 0.2 * criterion(scores[1], pred_seg_post_unchange)
    aux_loss += 0.2 * criterion(scores[2], pred_seg_pre_unchange)
    aux_loss += 0.2 * criterion(scores[1], labels[1])
    aux_loss += 0.2 * criterion(scores[2], labels[2])
