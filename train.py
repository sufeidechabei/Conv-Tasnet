import os
import pprint
import argparse
import random

from trainer import SiSnrTrainer
from dataset import make_dataloader, Dataset, ConeData

from conv_tasnet import TasNet

from conf import trainer_conf, nnet_conf, train_data, dev_data, chunk_size, num_spks
from utils import get_logger


from deepbeam import OnlineSimulationDataset, simulation_config, vctk_audio, truncator, ms_snsd, simulation_config_test


logger = get_logger(__name__)


def run(args):
    gpuids = tuple(map(int, args.gpus.split(",")))

    nnet = TasNet()
    trainer = SiSnrTrainer(nnet,
                           gpuid=gpuids,
                           checkpoint=args.checkpoint,
                           resume=args.resume,
                           **trainer_conf)

    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=args.batch_size,
                                   chunk_size=chunk_size,
                                   num_workers=args.num_workers, online=True, cone=False)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=args.batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=args.num_workers, online=True, cone=False)


    #dataset = ConeData(dev_data['data_path'], num_spks)
    dataset = OnlineSimulationDataset(vctk_audio, ms_snsd, 500, simulation_config_test, truncator, None, 50)

    trainer.run(train_loader, dev_loader, num_epochs=args.epochs, dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start ConvTasNet training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",
                        type=str,
                        default="0, 1, 2, 3",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--epochs",
                        type=int,
                        default=600,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        default="new",
                        help="Directory to dump models")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Number of utterances in each batch")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of workers used in data loader")
    parser.add_argument("--train_seed", type=str, default="4")
    parser.add_argument("--num_train", type=int, default=7000)
    parser.add_argument("--num_test", type=int, default=1000)
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    run(args)