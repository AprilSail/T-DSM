import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging


class BaseModuleTKG(nn.Module):
    def __init__(self):
        super(BaseModuleTKG, self).__init__()

    def score(self, src, rel, dst, time):
        raise NotImplementedError

    def dist(self, src, rel, dst, time):
        raise NotImplementedError

    def prob_logit(self, src, rel, dst, time):
        raise NotImplementedError

    def prob(self, src, rel, dst, time):
        return nnf.softmax(self.prob_logit(src, rel, dst, time))

    def constraint(self):
        pass

    def pair_loss(self, src, rel, dst, time, src_bad, dst_bad):
        # print(src.shape, rel.shape, dst.shape, time.shape, src_bad.shape, dst_bad.shape)
        d_good = self.dist(src, rel, dst, time)
        d_bad = self.dist(src_bad, rel, dst_bad, time)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, src, rel, dst, time, truth):
        probs = self.prob(src, rel, dst, time)
        n = probs.size(0)
        truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).cuda(), truth] + 1e-30)
        return -truth_probs


class BaseModelTKG(object):
    def __init__(self):
        self.mdl = None # type: BaseModuleTKG
        self.weight_decay = 0

    def save(self, filename):
        torch.save(self.mdl.state_dict(), filename)

    def load(self, filename):
        self.mdl.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))

    def gen_step(self, src, rel, dst, time, n_sample=1, temperature=1.0, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        n, m = dst.size()
        rel_var = Variable(rel.cuda())
        src_var = Variable(src.cuda())
        dst_var = Variable(dst.cuda())
        time_var = Variable(time.cuda())

        logits = self.mdl.prob_logit(src_var, rel_var, dst_var, time_var) / temperature
        probs = nnf.softmax(logits)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_srcs = src[row_idx, sample_idx.data.cpu()] ###
        sample_dsts = dst[row_idx, sample_idx.data.cpu()] ###
        sample_times = time[row_idx, sample_idx.data.cpu()] ###
        rewards = yield sample_srcs, sample_dsts
        if train:
            self.mdl.zero_grad()
            log_probs = nnf.log_softmax(logits)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
            reinforce_loss.backward()
            self.opt.step()
            self.mdl.constraint()
        yield None

    def dis_step(self, src, rel, dst, time, src_fake, dst_fake, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        src_var = Variable(src.cuda())
        rel_var = Variable(rel.cuda())
        dst_var = Variable(dst.cuda())
        time_var = Variable(time.cuda())
        src_fake_var = Variable(src_fake.cuda())
        dst_fake_var = Variable(dst_fake.cuda())
        losses = self.mdl.pair_loss(src_var, rel_var, dst_var, time_var, src_fake_var, dst_fake_var)
        fake_scores = self.mdl.fkscore(src_fake_var, rel_var, dst_fake_var, time_var)
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data

    def test_link(self, test_data, n_ent, heads, tails, filt=True):
        mrr_tot = 0
        mr_tot = 0
        hit1_tot = 0
        hit3_tot = 0
        hit10_tot = 0
        count = 0
        for batch_s, batch_r, batch_t, batch_time in batch_by_size(config().test_batch_size, *test_data):
            batch_size = batch_s.size(0)
            rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
            src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
            dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
            time_var = Variable(batch_time.unsqueeze(1).expand(batch_size, n_ent).cuda())
            all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                               .type(torch.LongTensor).cuda(), volatile=True)
            batch_dst_scores = self.mdl.score(src_var, rel_var, all_var, time_var).data
            batch_src_scores = self.mdl.score(all_var, rel_var, dst_var, time_var).data
            for s, r, t, time, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_time, batch_dst_scores, batch_src_scores):
                s = s.item()
                r = r.item()
                t = t.item()
                time = time.item()
                if filt:
                    if tails[(s, r, time)]._nnz() > 1:
                        tmp = torch.sum(dst_scores[t])
                        dst_scores += tails[(s, r, time)].cuda() * 1e30
                        dst_scores[t] = tmp
                    if heads[(t, r, time)]._nnz() > 1:
                        tmp = torch.sum(src_scores[s])
                        src_scores += heads[(t, r, time)].cuda() * 1e30
                        src_scores[s] = tmp
                mrr, mr, hit1, hit3, hit10 = mrr_mr_hitk(dst_scores, t)
                mrr_tot += mrr
                mr_tot += mr
                hit1_tot += hit1
                hit3_tot += hit3
                hit10_tot += hit10
                mrr, mr, hit1, hit3, hit10 = mrr_mr_hitk(src_scores, s)
                mrr_tot += mrr
                mr_tot += mr
                hit1_tot += hit1
                hit3_tot += hit3
                hit10_tot += hit10
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@1=%f, Test_H@3=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count, hit1_tot / count, hit3_tot / count, hit10_tot / count)
        return mrr_tot / count


    def train_with_ttt(self, test_data, n_ent, heads, tails, lstm_pred, filt=True):
        mseloss = torch.nn.MSELoss()
        # YAGO TTransE 3 0.00001
        n_epoch = 2
        best_loss = 999999
        self.mdl.train()
        for epoch in range(n_epoch):
            self.mdl.zero_grad()
            total_loss = 0.0
            if not hasattr(self, 'opt'):
                self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay, lr=0.000001)
            for batch_s, batch_r, batch_t, batch_time in batch_by_size(100, *test_data):
                # YAGO 183 GDELT 2592
                now_time = int(batch_time[0] - 2592)
                batch_size = batch_s.size(0)
                rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
                src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
                dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
                time_var = Variable(batch_time.unsqueeze(1).expand(batch_size, n_ent).cuda())
                all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                                   .type(torch.LongTensor).cuda(), volatile=True)
                batch_dst_scores = self.mdl.score(src_var, rel_var, all_var, time_var)
                batch_src_scores = self.mdl.score(all_var, rel_var, dst_var, time_var)
                batch_dst_scores = torch.nn.functional.softmax(batch_dst_scores, dim=1)
                batch_dst_scores = torch.sum(batch_dst_scores, dim=0)
                batch_src_scores = torch.nn.functional.softmax(batch_src_scores, dim=1)
                batch_src_scores = torch.sum(batch_src_scores, dim=0)

                f_p = torch.tensor(lstm_pred[now_time], dtype=torch.float32).cuda()
                loss_lstm_dst = mseloss(batch_dst_scores, f_p)
                loss_lstm_src = mseloss(batch_src_scores, f_p)
                losses = loss_lstm_dst + loss_lstm_src
                losses.backward()
                self.opt.step()
                # self.mdl.constraint()
                total_loss += losses.data
            logging.info('Epoch %d/%d, TTT_loss=%f', epoch + 1, n_epoch, total_loss)

