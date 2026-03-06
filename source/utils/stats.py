import os
import soundfile as sf
from collections import Counter
from source.utils.helpers import to_hms
from source.utils.translator import T

def generate_post_pipeline_stats(final_audio_dir, supported_exts, console, logger):
    all_files = os.listdir(final_audio_dir)
    audio_files = []
    for f in all_files:
        if any(f.lower().endswith(ext.lower()) for ext in supported_exts):
            audio_files.append(os.path.join(final_audio_dir, f))

    if not audio_files:
        return

    durations = []
    for path in audio_files:
        try:
            info = sf.info(path)
            durations.append(info.duration)
        except Exception as e:
            logger.debug(f"Stats: Soundfile failed on {os.path.basename(path)}: {e}")
            continue

    if not durations:
        logger.warning("No valid durations extracted via soundfile.")
        return

    total_sec = sum(durations)
    mean_sec = total_sec / len(durations)
    
    counts = Counter([round(d) for d in durations])
    sorted_keys = sorted(counts.keys())
    max_count = max(counts.values())

    console.separator()
    logger.info(T.translate("stats_title"))
    logger.info(T.translate("stats_total_duration", duration=to_hms(total_sec)))
    logger.info(T.translate("stats_mean_duration", mean=mean_sec))
    logger.info(T.translate("stats_distribution"))

    bar_width = 25 
    for sec in sorted_keys:
        count = counts[sec]
        bar = "[y]█[z]" * int((count / max_count) * bar_width)
        logger.info(f"    {sec:2}s: {bar} [c]{count}[z]")