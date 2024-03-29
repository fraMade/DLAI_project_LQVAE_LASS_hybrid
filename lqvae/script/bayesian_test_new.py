import numpy as np
import torch as t
import torch
import torchaudio
import museval
import os

import json
from tqdm import tqdm
from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior
from jukebox.transformer.ops import filter_logits
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.logger import def_tqdm
from jukebox.utils.sample_utils import get_starts
from jukebox.utils.torch_utils import empty_cache
from jukebox.data.files_dataset import FilesAudioDataset
from torch.utils.data.distributed import DistributedSampler
from jukebox.utils.audio_utils import audio_preprocess, audio_postprocess
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from pathlib import Path

from datasets_lass import ChunkedPairsDataset, SeparationSubset
from utils_lass import  get_dataset_subsample #, get_raw_to_tokens





rank, local_rank, device = setup_dist_from_mpi(port=29524)
from timeit import default_timer as timer

def make_models(vqvae_path, priors_list, sample_length, downs_t, sample_rate,
                levels=3, level=2, fp16=True, device='cuda'):
    # construct openai vqvae and priors
    vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=sample_length, downs_t=downs_t, sr=sample_rate,
                                                   restore_vqvae=vqvae_path)), device)
    prior_path_0 = priors_list[0]
    prior_path_1 = priors_list[1]

    prior_0 = make_prior(setup_hparams('small_prior', dict(levels=levels, level=level, labels=None,
                                                           restore_prior=prior_path_0, c_res=1, fp16_params=fp16,
                                                           n_ctx=8192)), vqvae, device)
    prior_1 = make_prior(setup_hparams('small_prior', dict(levels=levels, level=level, labels=None,
                                                           restore_prior=prior_path_1, c_res=1, fp16_params=fp16,
                                                           n_ctx=8192)), vqvae, device)
    priors = [prior_0, prior_1]
    return vqvae, priors

def create_mixture_from_audio_files(m1, m2, raw_to_tokens, sample_tokens,
                                    vqvae, save_path, sample_rate, alpha, device='cuda', shift=0.):
    #m1, _ = torchaudio.load(path_audio_1)
    #m2, _ = torchaudio.load(path_audio_2)

    # controllare che m1 e m2 devono essere di dimensioni (1, length) es (1, 5060608)

    shift = int(shift * sample_rate)
    assert sample_tokens * raw_to_tokens <= min(m1.shape[-1], m2.shape[-1]), "Sources must be longer than sample_tokens"
    minin = sample_tokens * raw_to_tokens
    
    m1_real    = m1[:, shift:shift+minin]
    m2_real    = m2[:, shift:shift+minin]
    mix        = alpha[0]*m1_real + alpha[1]*m2_real

    if not os.path.exists(f'{save_path}/'):
        os.makedirs(f'{save_path}/')

    torchaudio.save(f'{save_path}/real_mix.wav', mix.cpu().view(1,-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/real_bass.wav',  m1_real.cpu().view(1,-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/real_drums.wav',  m2_real.cpu().view(1,-1), sample_rate=sample_rate)

    # vqvae.encode(x.unsqueeze(-1).to(device), level, level + 1)[-1].squeeze(0).tolist()
    print(f"m1_real {m1_real.shape}")

    z_m1 = vqvae.encode(m1_real.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]
    z_m2 = vqvae.encode(m2_real.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]
    z_mixture = vqvae.encode(mix.unsqueeze(-1).to(device), start_level=2, bs_chunks=1)[0]

    latent_mix = vqvae.bottleneck.decode([z_mixture]*3)[-1]
    
    mix = vqvae.decode([z_mixture], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    m1 = vqvae.decode([z_m1], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    m2 = vqvae.decode([z_m2], start_level=2, bs_chunks=1).squeeze(-1)  # 1, 8192*128
    
    print(f"mix {mix.shape}")
    torchaudio.save(f'{save_path}/mix.wav', mix.cpu().view(1,-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/latent_bass.wav', m1.cpu().view(1,-1), sample_rate=sample_rate)
    torchaudio.save(f'{save_path}/latent_drums.wav', m2.cpu().view(1,-1), sample_rate=sample_rate)
    return mix, latent_mix, z_mixture, m1, m2, m1_real, m2_real

def sample_level(vqvae, priors, m, n_samples, n_ctx, hop_length, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0,
                 alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                 chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    xs_0 = torch.zeros(n_samples, 0, dtype=torch.long, device=device)
    xs_1 = torch.zeros(n_samples, 0, dtype=torch.long, device=device)

    if sample_tokens >= n_ctx:
        print("sample single windows")
        for start in get_starts(sample_tokens, n_ctx, hop_length):
            xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample_single_window(xs_0, xs_1, vqvae, priors, m, n_samples, n_ctx, start=start, sigma=sigma,
                                                                                            context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k,
                                                                                            top_p=top_p, bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                                                                            raw_to_tokens=raw_to_tokens, device=device, chunk_size=chunk_size,
                                                                                            latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
    else:
        print("Using ancestral sample")
        xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = ancestral_sample(vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma,
                                                                                    context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p,
                                                                                    bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                                                                    raw_to_tokens=raw_to_tokens, device=device, latent_loss=latent_loss,
                                                                                    top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
    return xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum

def rejection_sampling(nll0, nll1, res0, res1, remaining0, remaining1, m, alpha, bs, rejection_sigma, n_samples):
    nll_sum_0_sorted, indices_nll_sum_0_sorted = torch.sort(nll0)
    nll_sum_1_sorted, indices_nll_sum_1_sorted = torch.sort(nll1)

    global_likelihood = torch.zeros((bs, bs))
    global_posterior = torch.zeros((bs, bs))
    global_prior = torch.zeros((bs, bs))
    global_l2 = torch.zeros((bs, bs))

    for i in tqdm(range(bs)):
        for j in range(bs):
            global_prior[i, j] = (nll0[i] + nll1[j]) / n_samples # if drop else -np.inf
            global_l2[i, j] = torch.linalg.norm((alpha[0]*res0[i] +
                                                 alpha[1]*res1[j] - m), dim=-1)**2

    mp = global_prior.mean()
    ml = global_l2.mean()
    sigma_rejection_squared = - ml / (2*mp)
    print(f"sigma_rejection = {torch.sqrt(sigma_rejection_squared)}")
    global_likelihood = (-(1/(2.*(sigma_rejection_squared)))* global_l2)
    #if drop else 0.)
    # global_posterior[i, j] = global_likelihood[i, j] + global_prior[i, j]

    global_prior = global_prior.reshape(bs*bs)
    global_prior = torch.distributions.Categorical(logits=global_prior)
    global_prior_p = global_prior.probs.reshape(bs, bs)
    global_prior = global_prior.logits.reshape(bs, bs)

    global_posterior = global_prior + global_likelihood

    global_posterior = global_posterior.reshape(bs*bs) # n_samples, 2048 * 2048
    global_posterior = torch.distributions.Categorical(logits=global_posterior)
    global_posterior = global_posterior.probs.reshape(bs, bs)

    marginal_0 = global_posterior.sum(dim=-1)
    marginal_1 = global_posterior.sum(dim=0)
    marginal_1_sorted, marginal_1_idx_sorted = torch.sort(marginal_1)
    marginal_0_sorted, marginal_0_idx_sorted = torch.sort(marginal_0)

    global_l2_vectorized = global_l2.reshape(bs*bs)  # righe: prior_0, colonne: prior_1
    global_l2_topk_vectorized, global_l2_topk_idx_vectorized  = torch.topk(global_l2_vectorized, k=10, largest=False)

    rejection_index = torch.argmax(global_posterior)
    return (marginal_0_sorted, marginal_1_sorted, marginal_0_idx_sorted, marginal_1_idx_sorted,
            rejection_index // bs, rejection_index % bs)
   


def evaluate_sdr(gt0, gt1, res0, res1):
    sdr_0 = torch.zeros((res0.shape[0],))
    sdr_1 = torch.zeros((res1.shape[0],))
    for i in range(res0.shape[0]):
        sdr_0[i] = sdr(gt0.unsqueeze(-1).cpu().numpy(), res0[i, :].reshape(1, -1, 1).cpu().numpy())
        sdr_1[i] = sdr(gt1.unsqueeze(-1).cpu().numpy(), res1[i, :].reshape(1, -1, 1).cpu().numpy())

    sdr_0_sorted, sdr_0_sorted_idx = torch.sort(sdr_0)
    sdr_1_sorted, sdr_1_sorted_idx = torch.sort(sdr_1)
    return sdr_0, sdr_1, sdr_0_sorted, sdr_1_sorted, sdr_0_sorted_idx, sdr_1_sorted_idx


def sample_single_window(xs_0, xs_1, vqvae, priors, m, n_samples, n_ctx, start=0, sigma=0.01, context=8, fp16=False, temp=1.0,
                         alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                         chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    end = start + n_ctx
    # get z already sampled at current level
    x_0 = xs_0[:, start:end]
    x_1 = xs_1[:, start:end]

    sample_tokens = end - start
    conditioning_tokens, new_tokens = x_0.shape[1], sample_tokens - x_0.shape[1]

    if new_tokens <= 0:
        return xs_0, xs_1

    empty_cache()

    x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample(x_0, x_1, vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma, context=context,
                                                                    fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p, bs_chunks=bs_chunks,
                                                                    window_mode=window_mode, l_bins=l_bins, raw_to_tokens=raw_to_tokens, device=device,
                                                                    chunk_size=chunk_size, latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)

    # Update z with new sample
    x_0_new = x_0[:, -new_tokens:]
    x_1_new = x_1[:, -new_tokens:]

    xs_0 = torch.cat([xs_0, x_0_new], dim=1)
    xs_1 = torch.cat([xs_1, x_1_new], dim=1)

    return xs_0, xs_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum

def ancestral_sample(vqvae, priors, m, n_samples, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0, alpha=None,
                     top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128, device=None,
                     latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    x_cond = torch.zeros((n_samples, 1, priors[0].width), dtype=torch.float).to(device)
    xs_0, xs_1, x_0, x_1 = [], [], None, None

    log_p_0_sum = torch.zeros((n_samples,)).to(device)
    log_p_1_sum = torch.zeros((n_samples,)).to(device)
    # log_likelihood_sum = torch.zeros((n_samples,)).to(device)

    if latent_loss:
        codebook = torch.load("logs/sums/codebook_sum_precalc.pt")
        codebook = codebook.to(device)  # shape (1, 2048*2048)
        M = vqvae.bottleneck.one_level_decode(codebook)  # (1, 64, 2048*2048)
        M = M.squeeze(0)  # (64, 2048*2048)
        M = M.permute(1, 0)  # (2048*2048, 64)
        M = M.reshape(l_bins, l_bins, 64)  # (2048, 2048, 64)
        codebook = codebook.squeeze(0).reshape(l_bins, l_bins)
    else:
        tokens = torch.arange(l_bins).reshape(1, l_bins, 1).repeat(n_samples, 1, 1).to(device)

    for sample_t in def_tqdm(range(0, sample_tokens)):
        x_0, cond_0 = priors[0].get_emb(sample_t, n_samples, x_0, x_cond, y_cond=None)
        priors[0].transformer.check_cache(n_samples, sample_t, fp16)
        x_0 = priors[0].transformer(x_0, encoder_kv=None, sample=True, fp16=fp16) # Transformer
        
        if priors[0].add_cond_after_transformer:
            x_0 = x_0 + cond_0
        assert x_0.shape == (n_samples, 1, priors[0].width)
        x_0 = priors[0].x_out(x_0) # Predictions

        x_0 = x_0 / temp
        x_0 = filter_logits(x_0, top_k=top_k, top_p=top_p)
        p_0 = torch.distributions.Categorical(logits=x_0).probs # Sample and replace x
        log_p_0 = torch.log(p_0) # n_samples, 1, 2048

        x_1, cond_1 = priors[1].get_emb(sample_t, n_samples, x_1, x_cond, y_cond=None)
        priors[1].transformer.check_cache(n_samples, sample_t, fp16)
        x_1 = priors[1].transformer(x_1, encoder_kv=None, sample=True, fp16=fp16) # Transformer
        if priors[1].add_cond_after_transformer:
            x_1 = x_1 + cond_1
        x_1 = priors[1].x_out(x_1) # Predictions
        x_1 = x_1 / temp
        x_1 = filter_logits(x_1, top_k=top_k, top_p=top_p)
        p_1 = torch.distributions.Categorical(logits=x_1).probs # Sample and replace x
        log_p_1 = torch.log(p_1) # n_samples, 1, 2048

        log_p = log_p_0 + log_p_1.permute(0, 2, 1)  # n_samples, 2048, 2048 (p_1 sulle righe, p_0 sulle colonne)
        #print(f"{log_p.shape = }")

        ### START LOG LIKELIHOOD
        if not latent_loss:
            if window_mode == 'increment':
                x_0 = xs_0[sample_t // context * context: sample_t // context * context + sample_t % context]
                x_1 = xs_1[sample_t // context * context: sample_t // context * context + sample_t % context]
            elif window_mode == 'constant':
                x_0 = xs_0[sample_t - context: sample_t]
                x_1 = xs_1[sample_t - context: sample_t]
            else:
                raise NotImplementedError(f'Window mode {window_mode} not found')

            x_0 = torch.zeros(n_samples, 0).long().to(device) if x_0 == [] else torch.cat(x_0, dim=1)  # n_samples, t-1
            x_1 = torch.zeros(n_samples, 0).long().to(device) if x_1 == [] else torch.cat(x_1, dim=1)  # n_samples, t-1

            x_0 = torch.cat((x_0.unsqueeze(1).repeat(1, l_bins, 1), tokens), dim=-1).reshape(n_samples * l_bins, -1)  # n_samples * 2048, t
            x_1 = torch.cat((x_1.unsqueeze(1).repeat(1, l_bins, 1), tokens), dim=-1).reshape(n_samples * l_bins, -1)

            # n_samples * 2048, t-1 * 128, 1
            # n_samples * 2048, t-1 * 128
            # n_samples * 2048, 128
            d1 = vqvae.decode([x_0], start_level=2, bs_chunks=bs_chunks).squeeze(-1)[:, -raw_to_tokens:]
            d2 = vqvae.decode([x_1], start_level=2, bs_chunks=bs_chunks).squeeze(-1)[:, -raw_to_tokens:]

            # n_samples, 2048, 2048, 128
            y_mean = alpha[0]*d2.reshape(n_samples, l_bins, 1, raw_to_tokens) + alpha[1]*d1.reshape(n_samples, 1, l_bins, raw_to_tokens)

            m_t = m[:, (sample_t)*raw_to_tokens:(sample_t+1)*raw_to_tokens].reshape(1, 1, 1, -1)
            log_likelihood = -(1/(2.*(sigma**2)))*torch.linalg.norm(y_mean - m_t, dim=-1)**2  # n_samples, 2048, 2048

        else:
            # M: 2048, 2048, 64
            # m: 1, 1, 1, 64
            # log_likehood: 2048, 2048, 1
            # log_p: N, 2048, 2048, 1

            if delta_likelihood:
                factors = (codebook == m[:, sample_t]).nonzero()
                #log_likelihood = -1/(torch.zeros((n_samples, 2048, 2048)) + 0.0000000000000001).cuda()
                prop_fact = torch.tensor(0.9999).cuda()
                n_factors = factors.shape[0]
                # print(n_factors)
                log_likelihood = torch.log((1-prop_fact)/(2048*2048 - n_factors))*torch.ones((n_samples, 2048, 2048)).cuda()
                log_likelihood[:, factors[0], factors[1]] = torch.log(prop_fact/(factors.shape[0])) #torch.log(torch.tensor(1/factors.shape[0]).cuda())
            else:
                #l2 = torch.linalg.norm(M - m[:, :, sample_t].unsqueeze(0), dim=-1)**2 #attivare per multi sigma

                #mp = log_p.mean(-1).mean(-1)
                #ml = l2.mean(-1).mean(-1)
                #sigma_rejection_squared = - ml / (2*mp)
                #print(f"sigma_rejection = {torch.sqrt(sigma_rejection_squared)}")
                #log_likelihood = (-(1/(2.*(sigma.reshape(-1, 1, 1)))) * l2.unsqueeze(0).repeat(bs, 1, 1)) #attivare per multisigma

                log_likelihood = -(1/(2.*(sigma**2)))*torch.linalg.norm(M - m[:, :, sample_t].unsqueeze(0), dim=-1)**2  # n_samples, 2048, 2048

        # print(f"{log_likelihood.shape = }")
        # print(f"{torch.min(log_likelihood) = }")
        # print(f"{torch.max(log_likelihood) = }")
        # print(f"{torch.min(log_p) = }")
        # print(f"{torch.max(log_p) = }")
        #### END LIKELIHOOD ####

        log_posterior = log_likelihood + log_p  # n_samples, 2048, 2048 #.unsqueeze(-1).repeat(32, 1, 1)

        # print(f"{torch.min(log_posterior) = }")
        # print(f"{torch.max(log_posterior) = }")
        log_posterior = log_posterior.reshape(n_samples, l_bins*l_bins) # n_samples, 2048 * 2048
        log_posterior = filter_logits(log_posterior.unsqueeze(1), top_k=top_k_posterior, top_p=0.0).squeeze(1)
        posterior = torch.distributions.Categorical(logits=log_posterior)
        # print(f"{torch.min(posterior.probs) = }")
        # print(f"{torch.max(posterior.probs) = }")

        x = posterior.sample() # n_samples
        x_0 = (x % l_bins).reshape(n_samples, -1) # n_samples, 1
        x_1 = (x // l_bins).reshape(n_samples, -1) # n_samples, 1
        # x_1 = torch.div(x, l_bins, rounding_mode='trunc').reshape(n_samples, -1)
        log_p_0_sum += log_p_0[range(x_0.shape[0]), :, x_0.squeeze(-1)].squeeze(-1)
        log_p_1_sum += log_p_1[range(x_1.shape[0]), :, x_1.squeeze(-1)].squeeze(-1)
        #  log_likelihood_sum += log_likelihood[x_1, x_0].squeeze(-1)

        xs_0.append(x_0.clone())
        xs_1.append(x_1.clone())

    del x_0
    del x_1
    priors[0].transformer.del_cache()
    priors[1].transformer.del_cache()

    x_0 = torch.cat(xs_0, dim=1) # n_samples, sample_tokens
    #print(f"{x_0.shape = }")
    x_0 = priors[0].postprocess(x_0, sample_tokens) # n_samples, sample_tokens
    #print(f"{x_0.shape = }")
    x_1 = torch.cat(xs_1, dim=1)
    x_1 = priors[1].postprocess(x_1, sample_tokens)
    return x_0, x_1, log_p_0_sum, log_p_1_sum, None #log_likelihood_sum

def sdr(track1, track2):
    sdr_metric = museval.evaluate(track1, track2)
    sdr_metric[0][sdr_metric[0] == np.inf] = np.nan
    return np.nanmedian(sdr_metric[0])

def sample(xs_0, xs_1, vqvae, priors, m, n_samples, sample_tokens, sigma=0.01, context=8, fp16=False, temp=1.0,
           alpha=None, top_k=0, top_p=0.0, bs_chunks=1, window_mode='constant', l_bins=2048, raw_to_tokens=128,
           device=None, chunk_size=None, latent_loss=True, top_k_posterior=0, delta_likelihood=False):
    no_past_context = (xs_0 is None or xs_0.shape[1] == 0 or xs_1 is None or xs_1.shape[1] == 0)
    with torch.no_grad():
        if no_past_context:
            x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = ancestral_sample(vqvae, priors, m, n_samples, sample_tokens=sample_tokens, sigma=sigma,
                                                                                      context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k, top_p=top_p,
                                                                                      bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
                                                                                      raw_to_tokens=raw_to_tokens, device=device, latent_loss=latent_loss,
                                                                                      top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
        else:
            #x_0, x_1 = primed_sample(xs_0, xs_1, vqvae, priors, m, n_samples, sample_tokens=sample_tokens,
            #                         sigma=sigma, context=context, fp16=fp16, temp=temp, alpha=alpha, top_k=top_k,
            #                         top_p=top_p, bs_chunks=bs_chunks, window_mode=window_mode, l_bins=l_bins,
            #                         raw_to_tokens=raw_to_tokens, device=device, chunk_size=chunk_size,
            #                         latent_loss=latent_loss, top_k_posterior=top_k_posterior, delta_likelihood=delta_likelihood)
            nll_sum_0 = None
            nll_sum_1 = None
            print("Error: no_past_context=False")
    return x_0, x_1, nll_sum_0, nll_sum_1, None


if __name__ == '__main__':


     
   


    raw_to_tokens = 64
    l_bins = 2048
    sample_tokens = 1024 #250 #1024 #250 #1024
    sample_length = raw_to_tokens * sample_tokens

    hps = setup_hparams("vqvae", {})
    downs_t = (2, 2, 2)
    commit = 1
    alpha = [0.5, 0.5]
    sample_rate = 22050
    min_duration = 11.90
    levels = 3
    level = 2
    fp16 = True
    labels = False

    SILENCE_THRESHOLD = 1.5e-5



    ### get models and data path
    bass_audio_files_dir=  'data/downsampled/test_sources/bass/'
    drums_audio_files_dir= 'data/downsampled/test_sources/drums/'

    restore_vqvae = "./logs/lq_vae/checkpoint_step_15400.pth.tar"

    priors_list = ['logs/prior_bass/checkpoint_latest.pth.tar',
                   'logs/prior_drums/checkpoint_latest.pth.tar']
    ###
    
    collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], 0)

    #vqvae = make_vqvae(setup_hparams('vqvae', dict(sample_length=sample_length, downs_t=downs_t, sr=sample_rate, commit=commit, restore_vqvae=restore_vqvae)), device)
    
    #### load models from the path
    vqvae, priors = make_models(restore_vqvae, priors_list, sample_length, downs_t, sample_rate,
                                levels=levels, level=level, fp16=fp16, device=device)
    ######

    # raw_to_tokens = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t)
    print(f"sample_tokens {sample_tokens}")
    dataset = ChunkedPairsDataset(
        instrument_1_audio_dir=bass_audio_files_dir,
        instrument_2_audio_dir=drums_audio_files_dir,
        sample_rate=sample_rate,
        max_chunk_size=raw_to_tokens * sample_tokens,
        min_chunk_size=raw_to_tokens,
    )
    

    # subsample the test dataset
    indices = get_dataset_subsample(len(dataset), 50, seed=0)
    subdataset = SeparationSubset(dataset, indices=indices)

    n_ctx = min(priors[0].n_ctx, priors[1].n_ctx)
    hop_length = n_ctx // 2
    #multi_sigma = torch.tensor([0.2,0.4,0.6,0.8]*8).cuda()
    sigma           = 0.4  # 316227766  #0.316227766 # 0.316227766 # was the best 0.004 both for filtered and none
    rejection_sigma = 0.06 #0.0625 #10000 #0.04 #0.02 #0.018  # 0. #0.02
    context         = 50   # quello che funziona: 10
    bs              = 64
    top_k_posterior = 0
    bs_chunks = 1
    chunk_size = 32
    window_mode = 'constant'
    latent_loss = True
    delta_likelihood = False
    save_path = Path('logs/results')
    use_posterior_argmax = True

    shuffle_dataset = False
    np.random.seed(0)
    NUMBER_OF_PAIRS = 50

    # dataloader1 = test_loader_drums
    # dataloader2 = test_loader_bass
    # dataloader_iterator = iter(dataloader2)


    ### metrics init
    total_sdr0_gt_rejection_marginal, total_sdr1_gt_rejection_marginal = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_real_rejection_marginal, total_sdr1_real_rejection_marginal = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_gt, total_sdr1_gt = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_real, total_sdr1_real = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

    total_sdr0_gt_rejection_posterior, total_sdr1_gt_rejection_posterior = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    total_sdr0_real_rejection_posterior, total_sdr1_real_rejection_posterior = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
    avg_time_sep = 0.
    avg_time_rej = 0.
    ###
    tested = 0
    pair = 0
    

    loader = DataLoader(subdataset, batch_size=1, num_workers=1)

    if shuffle_dataset:
        pass
    else:
        print(f"len dataset {len(dataset)}")
        n_elem=0
        for batch_idx, batch in enumerate(tqdm(loader)):
            chunk_path = save_path / f"{batch_idx}"
            if chunk_path.exists():
                print(f"Skipping path: {chunk_path}")
                continue
                
            # track_1, track_2 = batch
            m1, m2 = batch
            m1 = m1.view(1,-1).cuda()
            m2 = m2.view(1,-1).cuda()
            print(f"raw chunk shape {m1.shape}")
            tested = 0
                    
            print("ready to start available chunks")
                        
            start_sep = timer()

            # m1 = track_1[0][batch_idx * sample_length: batch_idx * sample_length + sample_length].unsqueeze(0).cuda()
            # m2 = track_2[0][batch_idx * sample_length: batch_idx * sample_length + sample_length].unsqueeze(0).cuda()

            print(f'track {n_elem+1}: {tested+1}/{NUMBER_OF_PAIRS} (current_i: {batch_idx})')

            # pair_path = save_path/ str(pair)

            chunk_path = save_path / str(batch_idx)

            mix, latent_mix, z_mixture, m1, m2, m1_real, m2_real = create_mixture_from_audio_files(m1, m2,
                                                                                                    raw_to_tokens,
                                                                                                    sample_tokens,
                                                                                                    vqvae,
                                                                                                    chunk_path,
                                                                                                    sample_rate,
                                                                                                    alpha
                                                                                                    )
            m = latent_mix

            print(f"m shape {m.shape}")
            x_0, x_1, log_p_0_sum, log_p_1_sum, log_likelihood_sum = sample_level(vqvae,
                                                                                    [priors[0].prior,
                                                                                    priors[1].prior],
                                                                                    m=m, n_ctx=n_ctx,
                                                                                    hop_length=hop_length,
                                                                                    alpha=alpha,
                                                                                    n_samples=bs,
                                                                                    sample_tokens=sample_tokens,
                                                                                    sigma=sigma,
                                                                                    context=context,
                                                                                    fp16=fp16,
                                                                                    bs_chunks=bs_chunks,
                                                                                    window_mode=window_mode,
                                                                                    l_bins=l_bins,
                                                                                    raw_to_tokens=raw_to_tokens,
                                                                                    device=device,
                                                                                    chunk_size=chunk_size,
                                                                                    latent_loss=latent_loss,
                                                                                    top_k_posterior=top_k_posterior,
                                                                                    delta_likelihood=delta_likelihood)

            print(f"x0_shape {x_0.shape}")
            res_0 = vqvae.decode([x_0], start_level=2, bs_chunks=1).squeeze(-1)  # n_samples, sample_tokens*128
            res_1 = vqvae.decode([x_1], start_level=2, bs_chunks=1).squeeze(-1)  # n_samples, sample_tokens*128
            
            print(f"res_o shape{res_0.shape}")

            end_sep = timer()
            sep_time = end_sep - start_sep

            start_rej = timer()

            remaining_0 = None
            remaining_1 = None
            # noinspection PyTypeChecker
            marginal_0, marginal_1, marginal_0_idx_sorted, marginal_1_idx_sorted, argmax_posterior_0, argmax_posterior_1 = rejection_sampling(
                log_p_0_sum, log_p_1_sum, res_0, res_1, remaining_0,
                remaining_1, mix, alpha, bs,
                rejection_sigma=rejection_sigma, n_samples=sample_tokens)
            end_rej = timer()
            rej_time = end_rej - start_rej
                        # rejection_sampling_latent(log_p_0_sum, log_p_1_sum, log_likelihood_sum, bs)
            
            sdr0, sdr1, sdr0_sorted, sdr1_sorted,idx0_latent, idx1_latent = evaluate_sdr(m1, m2, res_0, res_1)
            sdr0_real, sdr1_real, sdr0_real_sorted, sdr1_real_sorted, idx0, idx1 = evaluate_sdr(m1_real, m2_real, res_0, res_1)

            torchaudio.save(f'{chunk_path}/res_drums.wav', res_0[idx0[-1]].cpu().view(1,-1), sample_rate=sample_rate)
            torchaudio.save(f'{chunk_path}/res_bass.wav', res_1[idx1[-1]].cpu().view(1,-1), sample_rate=sample_rate)
         

            print(f" best at idx {idx0[-1]} , {idx1[-1]}")
            print(f" best sdr0 {sdr0_real_sorted} best sdr1 {sdr1_real_sorted}")
            data = {"sdr0":sdr0_real.tolist(),"sdr1":sdr1_real.tolist(),"sdr0_sorted":sdr0_real_sorted.tolist(),"sdr1_sorted":sdr1_real_sorted.tolist()}
            srd_path = chunk_path / "sdr_results.json"

            with srd_path.open("w") as f:
                json.dump(data, f)
                
            
                    

            marginal_idx0 = marginal_0_idx_sorted[-1]
            marginal_idx1 = marginal_1_idx_sorted[-1]

            if (sdr0[argmax_posterior_0].isnan() or
                sdr1[argmax_posterior_1].isnan() or
                sdr0_real[argmax_posterior_0].isnan() or
                sdr1_real[argmax_posterior_1].isnan() or

                sdr0[marginal_idx0].isnan() or
                sdr1[marginal_idx1].isnan() or
                sdr0_real[marginal_idx0].isnan() or
                sdr1_real[marginal_idx1].isnan() or

                sdr0_sorted[-1].isnan() or
                sdr1_sorted[-1].isnan() or
                sdr0_real_sorted[-1].isnan() or
                sdr1_real_sorted[-1].isnan()):
                continue
        


            total_sdr0_gt_rejection_marginal += sdr0[marginal_idx0]  # sdr0[selected_marginal]
            total_sdr1_gt_rejection_marginal += sdr1[marginal_idx1]  # sdr1[selected_marginal]

            total_sdr0_real_rejection_marginal += sdr0_real[marginal_idx0]  # sdr0_real[selected_marginal]
            total_sdr1_real_rejection_marginal += sdr1_real[marginal_idx1]  # sdr1_real[selected_marginal]

            total_sdr0_gt_rejection_posterior += sdr0[argmax_posterior_0]  # sdr0[selected_marginal]
            total_sdr1_gt_rejection_posterior += sdr1[argmax_posterior_1]  # sdr1[selected_marginal]

            total_sdr0_real_rejection_posterior += sdr0_real[argmax_posterior_0]  # sdr0_real[selected_marginal]
            total_sdr1_real_rejection_posterior += sdr1_real[argmax_posterior_1]  # sdr1_real[selected_marginal]

            total_sdr0_gt += sdr0_sorted[-1]
            total_sdr1_gt += sdr1_sorted[-1]

            total_sdr0_real += sdr0_real_sorted[-1]
            total_sdr1_real += sdr1_real_sorted[-1]

            avg_time_sep += sep_time
            avg_time_rej += rej_time

            tested += 1

            print(f"sdr0_gt_rejection_marginal: {total_sdr0_gt_rejection_marginal / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr1_gt_rejection_marginal: {total_sdr1_gt_rejection_marginal / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr0_real_rejection_marginal: {total_sdr0_real_rejection_marginal / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr1_real_rejection_marginal: {total_sdr1_real_rejection_marginal / (n_elem * NUMBER_OF_PAIRS + tested)}")

            print(f"sdr0_gt_rejection_posterior: {total_sdr0_gt_rejection_posterior / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr1_gt_rejection_posterior: {total_sdr1_gt_rejection_posterior / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr0_real_rejection_posterior: {total_sdr0_real_rejection_posterior / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr1_real_rejection_posterior: {total_sdr1_real_rejection_posterior / (n_elem * NUMBER_OF_PAIRS + tested)}")

            print(f"sdr0_gt: {total_sdr0_gt / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr1_gt: {total_sdr1_gt / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr0_real: {total_sdr0_real / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"sdr1_real: {total_sdr1_real / (n_elem * NUMBER_OF_PAIRS + tested)}")

            print(f"avg_time_sep: {avg_time_sep / (n_elem * NUMBER_OF_PAIRS + tested)}")
            print(f"avg_time_rej: {avg_time_rej / (n_elem * NUMBER_OF_PAIRS + tested)}")

        
            if tested == NUMBER_OF_PAIRS:
                break
        n_elem += 1
    
    print(f"sdr0_gt_rejection_marginal: {total_sdr0_gt_rejection_marginal /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr1_gt_rejection_marginal: {total_sdr1_gt_rejection_marginal /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr0_real_rejection_marginal: {total_sdr0_gt_rejection_marginal /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr1_real_rejection_marginal: {total_sdr1_gt_rejection_marginal /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr0_gt_rejection_posterior: {total_sdr0_gt_rejection_posterior /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr1_gt_rejection_posterior: {total_sdr1_gt_rejection_posterior /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr0_real_rejection_posterior: {total_sdr0_gt_rejection_posterior /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr1_real_rejection_posterior: {total_sdr1_gt_rejection_posterior /  (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr0_gt: {total_sdr0_gt / (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr1_gt: {total_sdr1_gt / (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr0_real: {total_sdr0_real / (len(dataset)*NUMBER_OF_PAIRS)}")
    print(f"sdr1_real: {total_sdr1_real / (len(dataset)*NUMBER_OF_PAIRS)}")
