import os
import torch
import torchaudio
import pyloudnorm as pyln
from tqdm import tqdm

from source.utils.audio import get_audio_info
from source.utils.logger import get_logger
from source.utils.console import Console
from source.utils.helpers import to_hms
from source.utils.translator import T

logger = get_logger("Segmentation")
console = Console()

def normalize_lufs_tensor(tensor, sr, target_lufs):
    samples = tensor.numpy().T 
    meter = pyln.Meter(sr)
    try:
        current_lufs = meter.integrated_loudness(samples)
        normalized_samples = pyln.normalize.loudness(samples, current_lufs, target_lufs)
        return torch.from_numpy(normalized_samples.T).float()
    except Exception as e:
        logger.warning(T.translate("segmentation_lufs_failed", error=str(e)))
        return tensor

def process(input_file, output_dir, config, settings):
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(input_file)
    audio_meta, duration = get_audio_info(input_file)
    
    logger.info(T.translate("segmentation_analyzing", file=file_name))
    logger.info(T.translate("segmentation_audio_info", audio_meta=audio_meta))
    logger.info(T.translate("segmentation_duration", duration=to_hms(duration)))
    
    logger.info(T.translate("segmentation_loading_model"))
    #logger.info(T.translate("segmentation_waiting_message"))
    
    model, utils = torch.hub.load(
        repo_or_dir=settings.repo, 
        model=settings.model, 
        verbose=False
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils

    try:
        wav_vad = read_audio(input_file, sampling_rate=settings.internal_sampling_rate)
    except Exception as e:
        logger.error(T.translate("segmentation_read_error", error=str(e)))
        raise

    logger.info(T.translate("segmentation_detecting_timestamps"))
    #logger.info(T.translate("segmentation_waiting_message"))
    
    speech_timestamps = get_speech_timestamps(
        wav_vad, 
        model, 
        sampling_rate=settings.internal_sampling_rate, 
        threshold=config.threshold
    )

    logger.info(T.translate("segmentation_loading_hq"))
    audio_hq, sr_hq = torchaudio.load(input_file)
    
    is_mono_request = config.segment_format.channels == "mono"
    if is_mono_request and audio_hq.shape[0] > 1:
        status_mono = T.translate("status_mixed_to_mono")
        audio_hq = torch.mean(audio_hq, dim=0, keepdim=True)
    else:
        status_mono = T.translate("status_already_mono")
    logger.info(T.translate("segmentation_mono_mixed", status=status_mono))

    target_sr = config.segment_format.sampling_rate
    if sr_hq != target_sr:
        status_sr = T.translate("status_resampling", old=sr_hq, new=target_sr)
        resampler = torchaudio.transforms.Resample(sr_hq, target_sr)
        audio_hq = resampler(audio_hq)
        sr_hq = target_sr
    else:
        status_sr = T.translate("status_already_resampled", sr=sr_hq)
    logger.info(T.translate("segmentation_resampling", status=status_sr))

    logger.info(T.translate("segmentation_lufs_info", lufs=config.segment_format.loudness_lufs))

    segments_count = 0
    min_s = config.segment_file.duration.min_sec
    max_s = config.segment_file.duration.max_sec
    filename_template = config.segment_file.filename

    for idx, ts in enumerate(tqdm(speech_timestamps, desc=T.translate("segmentation_progress_bar"), leave=False)):
        start_sample = int(ts['start'] * (sr_hq / settings.internal_sampling_rate))
        end_sample = int(ts['end'] * (sr_hq / settings.internal_sampling_rate))
        duration_sec = (end_sample - start_sample) / sr_hq
        
        if min_s <= duration_sec <= max_s:
            chunk = audio_hq[:, start_sample:end_sample]
            chunk = normalize_lufs_tensor(chunk, sr_hq, config.segment_format.loudness_lufs)
            
            try:
                filename = filename_template.format(idx=idx)
            except (KeyError, ValueError):
                filename = f"segment_{idx:05d}.flac"

            out_path = os.path.join(output_dir, filename)
            torchaudio.save(
                out_path, 
                chunk, 
                sr_hq, 
                bits_per_sample=config.segment_format.bits_per_sample, 
                compression=0
            )
            segments_count += 1

    logger.info(T.translate("segmentation_done", kept=segments_count, total=len(speech_timestamps)))
    return segments_count