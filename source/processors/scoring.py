import os
import json
from tqdm import tqdm
from collections import Counter

from source.common.engine import InferenceEngine
from source.utils.logger import get_logger
from source.utils.translator import T

logger = get_logger("Scoring")

def get_lexical_complexity(text: str) -> float:
    words = text.lower().split()
    if not words: 
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    unique_ratio = len(set(words)) / len(words)
    return avg_len * unique_ratio

def process(segment_dir: str, config, engine_cfg, seg_config, settings, language: str):
    """
    Step 2: Scoring statistique (ASR + Lexical).
    """
    engine = InferenceEngine(config=engine_cfg)

    # Récupération de l'extension dynamique
    ext = seg_config.segment_format.extension.strip('.')
    audio_files = [f for f in os.listdir(segment_dir) if f.endswith(f'.{ext}')]
    
    if not audio_files:
        logger.warning(T.translate("scoring_no_audio", dir=segment_dir, ext=ext))
        return 0

    temp_data = []
    first_word_counts = Counter()

    # Log simplifié : focus sur le threshold
    logger.info(T.translate("scoring_stat_start", thresh=config.threshold))

    # Chargement du modèle ASR (Le message "Loading ASR Engine" est géré dans InferenceEngine)
    engine.ensure_asr_loaded()
    
    for filename in tqdm(audio_files, desc=T.translate("scoring_progress_bar"), leave=False):
        abs_path = os.path.abspath(os.path.join(segment_dir, filename))
        
        # Transcription via Qwen-ASR
        text = engine.transcribe(abs_path, language=language)
        
        if not text:
            logger.debug(T.translate("scoring_asr_error", file=filename))
            continue

        score = get_lexical_complexity(text)
        words = text.split()
        first_word = words[0].lower() if words else ""
        
        keep = True
        if score < config.threshold:
            keep = False
            logger.debug(T.translate("scoring_rejected_score", score=round(score, 2), thresh=config.threshold, file=filename))
        
        if keep and first_word and first_word_counts[first_word] >= config.max_redundancy:
            keep = False
            logger.debug(T.translate("scoring_rejected_redo", word=first_word, file=filename))

        if keep:
            if first_word:
                first_word_counts[first_word] += 1
            
            temp_data.append({
                "filename": filename,
                "abs_path": abs_path,
                "text": text,
                "score": round(score, 2)
            })

    # Sauvegarde des métadonnées temporaires
    scored_filename = settings.files.scored_metadata
    output_meta = os.path.join(segment_dir, scored_filename)
    
    try:
        with open(output_meta, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=4)
        # On affiche uniquement le nom du fichier en vert pour la clarté
        logger.info(T.translate("scoring_save_success", path=scored_filename))
    except Exception as e:
        logger.error(T.translate("scoring_save_error", error=str(e)))

    # Résumé final harmonisé
    logger.info(T.translate("scoring_summary", kept=len(temp_data), total=len(audio_files)))
    
    return len(temp_data)