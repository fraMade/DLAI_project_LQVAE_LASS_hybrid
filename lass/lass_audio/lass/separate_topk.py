import abc
import functools
from pathlib import Path
from typing import Callable, List, Mapping

import diba
import numpy as np
import torch
import torchaudio
import tqdm
import os
import json
import museval
from diba.diba import Likelihood
from torch.utils.data import DataLoader
from diba.interfaces import SeparationPrior
from timeit import default_timer as timer

# from lass.datasets import SeparationDataset
# from lass.datasets import SeparationSubset
# from lass.diba_interfaces import JukeboxPrior, SparseLikelihood
# from lass.utils import assert_is_audio, decode_latent_codes, get_dataset_subsample, get_raw_to_tokens, setup_priors, setup_vqvae
# from lass.datasets import ChunkedPairsDataset
# from jukebox.utils.dist_utils import setup_dist_from_mpi


from lass.datasets import SeparationDataset
from lass.datasets import SeparationSubset
from lass.diba_interfaces import JukeboxPrior, SparseLikelihood
from lass.utils import assert_is_audio, decode_latent_codes, get_dataset_subsample, get_raw_to_tokens, setup_priors, setup_vqvae
from lass.datasets import ChunkedPairsDataset
from jukebox.utils.dist_utils import setup_dist_from_mpi

# audio_root = Path(__file__).parent.parent


class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def separate(mixture) -> Mapping[str, torch.Tensor]:
        ...


class BeamsearchSeparator(Separator):
    def __init__(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        priors: Mapping[str, SeparationPrior], 
        likelihood: Likelihood, 
        num_beams: int,
    ):
        super().__init__()
        self.likelihood = likelihood
        self.source_types = ["bass","drums"] #list(priors.keys())
        self.priors = priors #list(priors.values())
        self.num_beams = num_beams

        self.encode_fn = encode_fn #lambda x: vqvae.encode(x.unsqueeze(-1), vqvae_level, vqvae_level + 1).view(-1).tolist()
        self.decode_fn = decode_fn #lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=vqvae_level)

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> Mapping[str, torch.Tensor]:
        # convert signal to codes
        mixture_codes = self.encode_fn(mixture) 

        # separate mixture (x has shape [2, num. tokens])
        x_bass, x_drums = diba.fast_beamsearch_separation(
            priors=self.priors,
            likelihood=self.likelihood,
            mixture=mixture_codes,
            num_beams=self.num_beams,
        )
        # print(f"are resutls different :{torch.equal(x_bass, x_drums)}")
        # decode results
        # decoded = {source: [] for source in self.source_types}
        # for xi  in x_bass:
        #     decoded[self.source_types[0]].append(self.decode_fn(xi))
        # for xi  in x_drums:
        #     decoded[self.source_types[1]].append(self.decode_fn(xi))

        # return decoded
        # res0 = x_bass.cpu()
        # res1 = x_drums.cpu()
        # print(f"mixture shape{mixture.shape}")
        res_bass = torch.zeros((x_bass.shape[0], mixture.shape[1])) # 32768))# 65536))
        res_drums = torch.zeros((x_drums.shape[0], mixture.shape[1])) # 65536))
        for i in range(x_bass.shape[0]):
            res_bass[i] = self.decode_fn(x_bass[i])
            res_drums[i] = self.decode_fn(x_drums[i])
        # print(f"x0 {x.shape}, x1 {x1.shape}")
        # return x_bass, x_drums
        results = {}
        results["bass"] = res_bass #[self.decode_fn(xi) for xi in x_bass]
        results["drums"] = res_drums #[self.decode_fn(xi) for xi in x_drums]
        return results
        # return {source: self.decode_fn(xi) for source, xi in zip(self.source_types, x)}
        # return {source: self.decode_fn(xi) for source, xi in zip(self.source_types, x)} #self.decode_fn(x0[-1:]), self.decode_fn(x1[-1:]) #{source: self.decode_fn(xi) for source, xi in zip(self.source_types, x)}

    
class TopkSeparator(Separator):
    def __init__(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        priors: Mapping[str, SeparationPrior],
        likelihood: Likelihood, 
        num_samples: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        super().__init__()
        self.likelihood = likelihood
        self.source_types = ["bass","drums"] # list(priors)
        self.priors = priors #list(priors.values())
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k

        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def separate(self, mixture: torch.Tensor):
        mixture_codes = self.encode_fn(mixture) 

        x_0, x_1 = diba.fast_sampled_separation(
            priors=self.priors,
            likelihood=self.likelihood,  ### added self
            mixture=mixture_codes,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        # print(f"are resutls different :{torch.equal(x_0, x_1)}")
        # print(f"mixture shape{mixture.shape}")
        # print(f"x shape {x_0.shape}")
        res_bass = torch.zeros((x_0.shape[0], mixture.shape[1])) # 32768))# 65536))
        res_drums = torch.zeros((x_1.shape[0], mixture.shape[1])) # 65536))
        
        for i in range(x_0.shape[0]):
            res_bass[i] = self.decode_fn(x_0[i])
            res_drums[i] = self.decode_fn(x_1[i])

        # print(f"x0 {x.shape}, x1 {x1.shape}")
        # return x_bass, x_drums
        results = {}
        results["bass"] = res_bass #[self.decode_fn(xi) for xi in x_bass]
        results["drums"] = res_drums #[self.decode_fn(xi) for xi in x_drums]
        # decode results
        # dec_bs = 32
        # num_batches = int(np.ceil(self.num_samples / dec_bs))
        # res_0, res_1 = [None]*num_batches, [None]*num_batches

        # for i in range(num_batches):
        #     res_0[i] = self.decode_fn(x_0[i * dec_bs:(i + 1) * dec_bs])
        #     res_1[i] = self.decode_fn(x_1[i * dec_bs:(i + 1) * dec_bs])

        # res_0 = torch.cat(res_0, dim=0)
        # res_1 = torch.cat(res_1, dim=0)
        
        # select best
        # best_idx = (0.5 * res_bass + 0.5 * res_drums - mixture.view(1,-1)).norm(p=2, dim=-1).argmin()
        return  results
        # return  {source: self.decode_fn(xi) for source, xi in zip(self.source_types, [x_0[best_idx], x_1[best_idx]])}


# -----------------------------------------------------------------------------

def sdr(track1, track2):
    sdr_metric = museval.evaluate(track1, track2)
    sdr_metric[0][sdr_metric[0] == np.inf] = np.nan
    return np.nanmedian(sdr_metric[0])

def evaluate_sdr(gt0, gt1, res0, res1):
    sdr_0 = torch.zeros((res0.shape[0],))
    sdr_1 = torch.zeros((res1.shape[0],))
    for i in range(res0.shape[0]):
        sdr_0[i] = sdr(gt0.unsqueeze(-1).cpu().numpy(), res0[i, :].reshape(1, -1, 1).cpu().numpy())
        sdr_1[i] = sdr(gt1.unsqueeze(-1).cpu().numpy(), res1[i, :].reshape(1, -1, 1).cpu().numpy())

    sdr_0_sorted, sdr_0_sorted_idx = torch.sort(sdr_0)
    sdr_1_sorted, sdr_1_sorted_idx = torch.sort(sdr_1)
    return sdr_0, sdr_1, sdr_0_sorted, sdr_1_sorted, sdr_0_sorted_idx, sdr_1_sorted_idx

@torch.no_grad()
def separate_dataset(
    dataset: SeparationDataset,
    separator: Separator,
    save_path: str,
    num_workers: int = 0,
):
    # convert paths
    save_path = Path(save_path)

    # get samples
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    # main loop
    # save_path.mkdir(exist_ok=True)
    stop_at=0
    
    total_sdr0_real = 0
    total_sdr1_real = 0
    for batch_idx, batch in enumerate(tqdm.tqdm(loader)):
        chunk_path = save_path / f"{batch_idx}"
        if chunk_path.exists():
            print(f"Skipping path: {chunk_path}")
            continue

        # load audio tracks
        origs = batch
        ori_0, ori_1 = origs
        print(f"chunk {batch_idx+1} out of {len(dataset)}")

        # generate mixture
        mixture = 0.5 * ori_0 + 0.5 * ori_1

        mixture = mixture.squeeze(0) # shape: [1 , sample-length]

        # Note: res0 and res1 have shape (n_samples, num_tokens)
        # res0, res1 
        results = separator.separate(mixture=mixture)

        
        chunk_path.mkdir(parents=True)

        # for sources_type, audio in results.items():
        #     results[sources_type] = torch.tensor(audio)

        # sdr_0_real, sdr_1_real, sdr_0_sorted_real, sdr_1_sorted_real, idx0_real, idx1_real = evaluate_sdr(ori_0.squeeze(0), ori_1.squeeze(0), results["bass"].view(1,-1), results["drums"].view(1,-1))
        sdr_0_real, sdr_1_real, sdr_0_sorted_real, sdr_1_sorted_real, idx0_real, idx1_real = evaluate_sdr(ori_0.squeeze(0), ori_1.squeeze(0), results["bass"], results["drums"])

        # print(f"SDR di track {batch_idx}:\nsdr_0:{sdr_0}\nsdr_1:{sdr_1}\nsdr_0_sorted: {sdr_0_sorted}\nsdr_1_sorted:{sdr_1_sorted}")#\n sep_time: {sep_time}")
        # print(f"Candidates {results['bass'].shape}, {results['drums'].shape} ")
        print(f"SDR di track {batch_idx}:\nsdr_0_sorted: {sdr_0_sorted_real[-1]}\nsdr_1_sorted:{sdr_1_sorted_real[-1]}")

        # print(ori_0.shape)
        # torchaudio.save(str(chunk_path / f"bass.wav"), ori_0.view(-1).cpu(), sample_rate=22050)
        # torchaudio.save(str(chunk_path / f"drums.wav"), ori_1.view(-1).cpu(), sample_rate=22050)
        # torchaudio.save(str(chunk_path / f"mixture.wav"), mixture.view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"bass.wav"), ori_0.view(1,-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"drums.wav"), ori_1.view(1,-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / f"mixture.wav"), mixture.view(1,-1).cpu(), sample_rate=22050)
        
        # for source_type, res in results.items():
        #     torchaudio.save(str(chunk_path / f"{source_type}.wav"), res.view(-1).cpu(), sample_rate=22050)
        
        best_bass = results["bass"][idx0_real[-1]]
        best_drums = results["drums"][idx1_real[-1]]
        # torchaudio.save(str(chunk_path / "best_bass.wav"), best_bass.view(-1).cpu(), sample_rate=22050)
        # torchaudio.save(str(chunk_path / "best_drums.wav"), best_drums.view(-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / "best_bass.wav"), best_bass.view(1,-1).cpu(), sample_rate=22050)
        torchaudio.save(str(chunk_path / "best_drums.wav"), best_drums.view(1,-1).cpu(), sample_rate=22050)

        data = {"sdr0_real":sdr_0_real.tolist(),"sdr1_real":sdr_1_real.tolist(),
                "sdr0_sorted_real":sdr_0_sorted_real.tolist(),"sdr1_sorted_real":sdr_1_sorted_real.tolist()}
        
        srd_path = os.path.join(chunk_path,'sdr_results.json')

        with open(srd_path, 'w') as f:
            json.dump(data, f)

        total_sdr0_real += sdr_0_sorted_real[-1]
        total_sdr1_real += sdr_1_sorted_real[-1]

        # torchaudio.save(str(chunk_path / f"sep_1.wav"), res1.view(-1).cpu(), sample_rate=22050)
        # for index, key in enumerate(seps):
        #     for i, song in enumerate(seps[key]):
        #         torchaudio.save(str(chunk_path / f"song_prior{index}_{i}.wav"), song.view(-1).cpu(), sample_rate=22050)
        # save separated audio
        # save_fn(
        #     separated_signals=[sep.unsqueeze(0) for sep in seps.values()],
        #     original_signals=[ori.squeeze(0) for ori in origs],
        #     path=chunk_path,
        # )
        # del res0, res1, origs
        
        stop_at += 1
        if stop_at == 25:
            break

    print(f"Average sdr0_real: {total_sdr0_real / (stop_at )}")
    print(f"Average sdr1_real: {total_sdr1_real / (stop_at )}")

    data = {"Average_sdr0_real":(total_sdr0_real.tolist() / (stop_at)),"Average_sdr1_real": (total_sdr1_real.tolist() / (stop_at))}
    
    srd_path = os.path.join(save_path,'Average_sdr_results.json')

    with open(srd_path, 'w') as f:
        json.dump(data, f)
# -----------------------------------------------------------------------------


def save_separation(
    separated_signals: List[torch.Tensor],
    original_signals: List[torch.Tensor],
    sample_rate: int,
    path: Path,
):
    assert_is_audio(*original_signals, *separated_signals)
    #assert original_1.shape == original_2.shape == separation_1.shape == separation_2.shape
    assert len(original_signals) == len(separated_signals)
    for i, (ori, sep) in enumerate(zip(original_signals, separated_signals)):
        torchaudio.save(str(path / f"ori{i+1}.wav"), ori.view(-1).cpu(), sample_rate=sample_rate)
        torchaudio.save(str(path / f"sep{i+1}.wav"), sep.view(-1).cpu(), sample_rate=sample_rate)


def main(
    
    
    audio_dir_1: str = "data/downsampled/test_sources/bass",#"./data/downsampled/test_sources/bass", #"./data/downsampled/test_sources/bass",
    audio_dir_2: str = "data/downsampled/test_sources/drums",#"./data/downsampled/test_sources/drums",
    vqvae_path: str =  "logs/vq_vae/checkpoint_step_15588.pth.tar", #"logs/lq_vae/checkpoint_step_15400.pth.tar",# #"./logs/lass-slakh-ckpts/vqvae.pth.tar", #, #"./logs/vq_vae/checkpoint_step_15588.pth.tar", #
    prior_1_path: str = "logs/prior_bass/checkpoint_latest_bass.pth.tar",  #  #"./logs/prior_bass/checkpoint_latest.pth.tar", , # /checkpoint_latest.pth.tar,
    prior_2_path: str = "logs/prior_drums/checkpoint_latest_drums.pth.tar",   # #./logs/prior_drums/checkpoint_latest.pth.tar,
   
    sum_frequencies_path: str = "./logs/vq_vae_sums/sum_dist_11000.npz", #"./logs/lass-slakh-ckpts/sum_frequencies.npz", , #

    vqvae_type: str = "vqvae",
    prior_1_type: str = "small_prior",
    prior_2_type: str = "small_prior",
    max_sample_tokens: int = 1024, #512, #250, #512, #1024, #512,#512,#1024,
    sample_rate: int = 22050, #44100, #22050,
    save_path: str = "./lass/results_subtest_vqvae_topk",
    resume: bool = False,
    num_pairs: int = 50, # forse da mettere 3 o 2
    seed: int = 0,
    **kwargs,
):
    # convert paths
    save_path = Path(save_path)
    audio_dir_1 = Path(audio_dir_1)
    audio_dir_2 = Path(audio_dir_2)

    #if not resume and save_path.exists():
    #    raise ValueError(f"Path {save_path} already exists!")

    rank, local_rank, device = setup_dist_from_mpi(port=29533, verbose=True)

    # setup models
    vqvae = setup_vqvae(
        vqvae_path=vqvae_path,
        vqvae_type=vqvae_type,
        sample_rate=sample_rate,
        sample_tokens=max_sample_tokens,
        device=device,
    )

    priors = setup_priors(
        prior_paths=[prior_1_path, prior_2_path],
        prior_types=[prior_1_type, prior_2_type],
        vqvae=vqvae,
        fp16=True,
        device=device,
    )

    # priors = {
    #     prior_1_path.split('/')[-2]: priors[0],
    #     prior_2_path.split('/')[-2]: priors[1],
    # }

    # create separator
    level = vqvae.levels - 1
    separator = TopkSeparator(
        encode_fn=lambda x: vqvae.encode(x.unsqueeze(-1).to(device), level, level + 1)[-1].squeeze(0).tolist(), # TODO: check if correct
        decode_fn=lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=level),
        priors = [JukeboxPrior(p.prior, torch.zeros((), dtype=torch.float32, device=device)) for p in priors],
        # priors={k:JukeboxPrior(p.prior, torch.zeros((), dtype=torch.float32, device=device)) for k,p in priors.items()},
        likelihood=SparseLikelihood(sum_frequencies_path, device, 3.0),
        num_samples=100,#100, #num_beams=100,
        **kwargs,
    )
    # separator = BeamsearchSeparator(
    #     encode_fn=lambda x: vqvae.encode(x.unsqueeze(-1).to(device), level, level + 1)[-1].squeeze(0).tolist(), # TODO: check if correct
    #     decode_fn=lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=level),
    #     priors = [JukeboxPrior(p.prior, torch.zeros((), dtype=torch.float32, device=device)) for p in priors],
    #     # priors={k:JukeboxPrior(p.prior, torch.zeros((), dtype=torch.float32, device=device)) for k,p in priors.items()},
    #     likelihood=SparseLikelihood(sum_frequencies_path, device, 3.0),
    #     num_beams=200,#100, #num_beams=100,
    #     **kwargs,
    # )

    # setup dataset
    raw_to_tokens = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t)
    dataset = ChunkedPairsDataset(
        instrument_1_audio_dir=audio_dir_1,
        instrument_2_audio_dir=audio_dir_2,
        sample_rate=sample_rate,
        max_chunk_size=raw_to_tokens * max_sample_tokens,
        min_chunk_size=raw_to_tokens,
    )
    

    # subsample the test dataset
    indices = get_dataset_subsample(len(dataset), num_pairs, seed=seed)
    subdataset = SeparationSubset(dataset, indices=indices)

    # separate subsample
    separate_dataset(
        dataset=subdataset,
        separator=separator,
        save_path=save_path,
        # save_fn=functools.partial(save_separation, sample_rate=sample_rate),
        # resume=resume,
    )

if __name__ == "__main__":
    main()