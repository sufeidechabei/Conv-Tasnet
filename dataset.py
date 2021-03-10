# wujian@2018
import random
import torch as th
import numpy as np
import os
import scipy.io.wavfile as wf

import torch.utils.data as dat
import librosa
import json
from utils import MAX_INT16
from deepbeam import OnlineSimulationDataset, simulation_config, vctk_audio, truncator,\
    ms_snsd, simulation_config_test
from torch.utils.data.dataloader import default_collate
from pathlib import Path
from torch.utils.data import DataLoader as DL


def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16,
                    online=False,
                    cone=False):
    if not online and not cone:
        dataset = Dataset(**data_kwargs)
    else:
        if cone:
            dataset = ConeData(**data_kwargs)
            #if train:
                #shuffle = True
            #else:
                #shuffle = False
            #return DL(dataset,
                      #batch_size=batch_size,
                      #shuffle=shuffle,
                      #num_workers=num_workers)
        else:
            dataset = OnlineData(train=train)
            #if train:
               #shuffle = True
            #else:
               #shuffle = False
        #return DL(dataset,
            #batch_size=batch_size,
            #shuffle=shuffle,
            #num_workers=num_workers)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


class Dataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, data_path, num_speakers):
        # self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        # self.ref = [
        # WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp
        all_dirs = os.listdir(data_path)
        # num_samples = len(all_dirs)
        # all_dirs = all_dirs[:int(num_samples/2)]
        if "oracle" in all_dirs:
            all_dirs.remove('oracle')
        if "no_oracle" in all_dirs:
            all_dirs.remove('no_oracle')
        self.data = []
        self.label = []
        for dir in all_dirs:
            self.data.append(wf.read(os.path.join(data_path, dir, 'mix.wav'))[1].astype(np.float)/MAX_INT16)
            speakers = []
            for i in range(num_speakers):
                speakers.append(os.path.join(data_path, dir, str(i) + "sub.wav"))
            speaker_data = []
            for speaker in speakers:
                spk_data = (wf.read(os.path.join(data_path, dir, speaker),
                                    )[1].astype(np.float)/MAX_INT16)[:, 0]
                speaker_data.append(np.expand_dims(spk_data, axis=0))
            self.label.append(np.concatenate(speaker_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # key = self.mix.index_keys[index]
        # mix = self.mix[key]
        # ref = [reader[key] for reader in self.ref]
        mix = self.data[index]
        ref = self.label[index][:, 0, :]
        return {
            "mix": mix.astype(np.float32),
            "ref": [r.astype(np.float32) for r in ref]
        }


class ConeData(object):
    def __init__(self, data_path, num_speakers):
        self.dirs = os.listdir(data_path)[:32]
        self.num_speakers = num_speakers
        self.data_path = data_path

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, item):
        """
                    This is a modified version of the SpatialAudioDataset DataLoader
        """
        curr_dir = os.path.join(self.data_path, self.dirs[item])
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            json_data = json.load(json_file)
        num_voices = self.num_speakers
        # mic_files = sorted(list(Path(curr_dir).rglob('*mixed.wav')))

        # All voice signals
        keys = ["voice{:02}".format(i) for i in range(num_voices)]

        # Comment out this line to do voice only, no bg
        if "bg" in json_data:
            keys.append("bg")
        """
        Loading the sources
        """
        # Iterate over different sources
        all_sources = []
        #target_voice_data = []
        #voice_positions = []
        for key in keys:
            gt_audio_files = sorted(list(Path(curr_dir).rglob("*" + key + ".wav")))
            assert (len(gt_audio_files) > 0)
            gt_waveforms = []

            # Iterate over different mics
            for _, gt_audio_file in enumerate(gt_audio_files):
                gt_waveform, _ = librosa.core.load(gt_audio_file,
                                                   mono=True)
                gt_waveforms.append(gt_waveform)

            single_source = np.stack(gt_waveforms)
            all_sources.append(single_source)
            #locs_voice = np.arctan2(json_data[key]["position"][1],
                                    #json_data[key]["position"][0])
            #voice_positions.append(locs_voice)
            #print(locs_voice)

        all_sources = np.stack(all_sources)/MAX_INT16  # n voices x n mics x n samples
        mixed_data = np.sum(all_sources, axis=0).T  # n mics x n samples
        all_sources = all_sources[:, 0, :]
        return {
            "mix": mixed_data.astype(np.float32),
            "ref": [r.astype(np.float32) for r in all_sources]
        }


class OnlineData(object):
    def __init__(self, train=True):
        if train:
            num_samples = 2000
            self.data = OnlineSimulationDataset(vctk_audio, ms_snsd, num_samples, simulation_config, truncator,
                                                None,
                                                100
                                                )
        else:
            num_samples = 500
            self.data = OnlineSimulationDataset(vctk_audio,
                                                ms_snsd,
                                                num_samples,
                                                simulation_config_test,
                                                truncator,
                                                None,
                                                50
                                                )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):
        # key = self.mix.index_keys[index]
        # mix = self.mix[key]
        # ref = [reader[key] for reader in self.ref]
        mix = self.data[index][0].T/MAX_INT16 #
        ref = self.data[index][3][..., :mix.shape[0]]/MAX_INT16 #n_speaker * channel * length
        return {
            "mix": mix.astype(np.float32),
            "ref": [r.astype(np.float32) for r in ref]
        }


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000):

        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):

        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[..., s:s + self.chunk_size] for ref in eg["ref"]]
        return chunk

    def split(self, eg):
        N = eg["mix"].shape[0]
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = [
                np.pad(ref, (0, P), "constant") for ref in eg["ref"]
            ]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):

    """
    Online dataloader for chunk-level PIT  ,
    """

    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):

            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj


class Audio(Dataset):
    def __init__(self, data_path, training=True, sr=44100, num_speakers=2):

        self.training = training
        all_dirs = os.listdir(data_path)
        # num_samples = len(all_dirs)
        # all_dirs = all_dirs[:int(num_samples/2)]
        if "oracle" in all_dirs:
            all_dirs.remove('oracle')
        if "no_oracle" in all_dirs:
            all_dirs.remove('no_oracle')
        self.data = []
        self.label = []
        for dir in all_dirs:
            self.data.append(wf.read(os.path.join(data_path, dir, 'mix.wav'))[1].astype(np.float).T)
            speakers = []
            for i in range(num_speakers):
                speakers.append(os.path.join(data_path, dir, str(i) + "sub.wav"))
            speaker_data = []
            for speaker in speakers:
                spk_data = (wf.read(os.path.join(data_path, dir, speaker),
                                                    )[1].T.astype(np.float))[0]
                speaker_data.append(np.expand_dims(spk_data, axis=0))
            self.label.append(np.concatenate(speaker_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


if __name__ == '__main__':
    pass