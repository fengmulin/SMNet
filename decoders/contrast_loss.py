import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ContrastLoss(nn.Module):

    def __init__(self,
                 hidden_dim = 128,
                 threshold = 0.3,
                 ):
        super(ContrastLoss, self).__init__()
        self.hidden_dim = hidden_dim
        self.tem=0.5
        self.threshold =threshold
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)
    def negloss(self, query,  neg_sets):
        query = query.reshape(-1, self.hidden_dim)
  
        # neg_sets = neg_sets.reshape(-1, self.hidden_dim)
        num_p = neg_sets.shape[0]
       
        Q_neg = F.cosine_similarity(query, neg_sets, dim=1)
        # Q_pos = F.cosine_similarity(query, pos_sets, dim=1)
        loss = 1/(1+torch.exp(-Q_neg*10))
        return loss.sum()/num_p
        # print(Q_neg,"****")
        loss =  Q_neg +1 
        num = min(Q_neg.shape[0],1000)
        loss, _ = torch.topk(loss,num)
        print(num, loss.max(),111,loss.mean())
        return loss.sum()/num
        return (torch.mean(Q_neg)+1)
    def INFOloss(self, query, pos_sets, neg_sets, tem):
        
        query = query.reshape(-1, self.hidden_dim)
        # pos_sets = pos_sets.reshape(-1, self.hidden_dim)
        # neg_sets = neg_sets.reshape(-1, self.hidden_dim)
        N = pos_sets.shape[0]
        Q_pos = F.cosine_similarity(query, pos_sets, dim=1)
        Q_neg = F.cosine_similarity(query, neg_sets, dim=1)
        Q_neg_exp_sum = torch.sum(torch.exp(Q_neg / tem))
        single_in_log = torch.exp(Q_pos / tem) / (torch.exp(Q_pos/tem) + Q_neg_exp_sum)
        batch_log = torch.sum(-1 * torch.log(single_in_log)) / N
        return batch_log
    def posloss(self, query, pos_sets):
    
        query = query.reshape(-1, self.hidden_dim)
        
        # print(query.shape,pos_sets.shape)
        # pos_sets = pos_sets.reshape(-1, self.hidden_dim)
        # num_p = pos_sets.shape[0]
        Q_pos = F.cosine_similarity(query, pos_sets, dim=1)
        loss = torch.exp(-Q_pos*10)
        # loss = - Q_pos +1 
        # print(loss.shape)
        # raise
        # num = min(Q_pos.shape[0],5000)
        # loss,_ = torch.topk(loss,num)
        # print(num, loss.max(),222,loss.mean())
        return loss.sum()
    def forward(self, fea_middle, pred, gt,mask, reduction='mean'):
        # if self.use_sigmoid:
        #     pred = torch.sigmoid(pred)
        # if ignore is not None:
        #     pred = pred * ignore
        #     target = target * ignore
        gt =  self.max_pool(gt[:, 0, :, :])
        pred =  self.max_pool(pred[:, 0, :, :])
        mask =  self.max_pool(mask)
        
        positive = (gt * mask)
        negative = ((1 - gt) * mask)
        pos_pred = positive*pred
        neg_pred = negative*pred
        # positive_count = int(positive.float().sum())
        # negative_count = min(int(negative.float().sum()),
        #                     int(positive_count * self.negative_ratio))
        # pos_index = pred>0.3
        # neg_pred = positive==1 & pred>=0.3
        # pos_pred = negative==1 & pred<=0.3

        # raise
        fea_middle = fea_middle.permute(0,2,3,1)
        # print(positive.shape, fea_middle.shape)
        # raise
        q_gt = fea_middle[positive==1]
        # print(q_gt.shape)
        q_gt = torch.mean(q_gt, dim=(0))
        # raise
        # q_gt = torch.mean(q_gt, dim=0) 
        # print(pos_pred.shape)
        # raise
        # print(type(positive==1),type(pred>0.3))


        fea_neg = fea_middle[ neg_pred>=0.2 ]
        # if fea_neg.shape[0]<1000:
        #     fea_neg = fea_middle[ neg_pred>=0.3 ]
        #fea_pos = fea_middle[ pos_pred<=0.3 ]
        # print(fea_pos.shape, q_gt.shape)
        # raise
        # if fea_pos.shape[0]<500:
        #     fea_pos = fea_middle[ pos_pred<=0.3 ]
        # print(fea_neg.shape, fea_pos.shape)
        # raise
        # loss = self.INFOloss(q_gt, fea_pos, fea_neg, self.tem)
        if fea_neg.shape[0]>0:
            loss_contrast_neg = self.negloss(q_gt,fea_neg)
        else:
            loss_contrast_neg = 0
        # if fea_pos.shape[0]>0:
        #     loss_contrast_pos = self.posloss(q_gt,fea_pos)
        # else:
        #     loss_contrast_pos = 0
        # # loss_contrast_hb = self.infoloss(q_easy_shrink,es_pos,hs_pos)
        # loss = loss_contrast_pos+loss_contrast_neg
        loss = loss_contrast_neg
        # if reduction == 'mean':
        #    loss = torch.mean(loss)
        return loss