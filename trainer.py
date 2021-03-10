import os

import time
import logging
import numpy as np
import sys
import copy
import torch as th
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from conv_tasnet_1 import Conv1D
from utils import compute_sdr, MAX_INT16, center_trim

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=20,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=20):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            nnet.encoder_1d = Conv1D(6, 256, 20, stride=10, padding=0)
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs)
        else:
            nnet.encoder_1d = Conv1D(6, 256, 20, stride=10, padding=0)
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)
        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()
            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def get_sdr(self, dataset):
        input_sdr_list = []
        output_sdr_list = []
        with th.no_grad():
            self.nnet.eval()
            for i in range(len(dataset)):
                input = dataset.__getitem__(i)
                mix = np.expand_dims(input[0], axis=0) / MAX_INT16 # 1 * channel * length
                ref = input[3] * MAX_INT16
                raw = th.tensor(mix, dtype=th.float32, device=self.gpuid[0])
                spks = self.nnet(raw)
                ref = th.tensor(ref, dtype=th.float32, device=self.gpuid[0])
                ref = center_trim(ref, spks).transpose(1, 0)
                _, spks = self.criterion(spks,
                                         ref, return_est=True)
                spks = spks.data.cpu().numpy().squeeze()
                ref = ref.data.cpu().numpy()

                for idx, samps in enumerate(spks):
                    samps = samps * MAX_INT16
                    input_sdr_list.append(compute_sdr(ref[0, idx], mix[0, 0, :]))
                    output_sdr_list.append(compute_sdr(ref[0, idx], samps))
            input_sdr_array = np.array(input_sdr_list)
            output_sdr_array = np.array(output_sdr_list)
            result = np.median(output_sdr_array - input_sdr_array)
            print("The SNR: " + str(result))
        return result

    def run(self, train_loader, dev_loader, num_epochs=50, dataset=None):
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # check if save is OK
            self.save_checkpoint(best=False)
            cv = self.eval(dev_loader)
            best_loss = cv["loss"]
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0
            # make sure not inf
            self.scheduler.best = best_loss
            best_sdr = 0
            best_model = None

            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                tr = self.train(train_loader)

                stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                    tr["loss"], tr["cost"], tr["batches"])
                cv = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                    cv["loss"], cv["cost"], cv["batches"])
                stats["scheduler"] = ""
                if cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv["loss"])
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                sdr = self.get_sdr(dataset)

                if best_sdr < sdr:
                    best_sdr = sdr
                    best_model = copy.deepcopy(self.nnet)

                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    print("The best sdr:"  + str(best_sdr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, num_epochs))
            th.save(best_model.state_dict(), os.path.join(self.checkpoint, "best.ckpt"))
            print("The final best sdr:" + str(best_sdr))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    @staticmethod
    def sisnr(x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        s = center_trim(s, x)
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, egs):
        # spks x n x S
        ests = th.nn.parallel.data_parallel(
            self.nnet, th.transpose(egs["mix"], 2, 1), device_ids=self.gpuid)
        # spks x n x S
        refs = th.stack(egs["ref"]).transpose(1, 0).squeeze()
        if isinstance(ests, list):
            ests = th.stack(ests)

        #refs = center_trim(refs, ests).reshape((refs.size()[0], refs.size(1), -1))
        #ests = ests.reshape((ests.size(0), ests.size(1), -1))
        return self.criterion(ests, refs)