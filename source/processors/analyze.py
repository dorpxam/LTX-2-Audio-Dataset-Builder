import os
import json
import traceback
from tqdm import tqdm

from source.common.engine import InferenceEngine
from source.utils.logger import get_logger
from source.utils.translator import T

logger = get_logger("Analyze")

def process(metadata_file: str, config, engine_cfg, settings_internals):
    """
    Step 3: Analyse sémantique via LLM.
    """
    if not os.path.exists(metadata_file):
        logger.error(T.translate("analyze_metadata_not_found", path=metadata_file))
        return 0

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(T.translate("analyze_json_read_error", error=str(e)))
        return 0

    if not data:
        logger.warning(T.translate("analyze_metadata_empty"))
        return 0

    # L'InferenceEngine gère le chargement propre du LLM
    engine = InferenceEngine(config=engine_cfg)

    final_selection = []
    min_score = config.min_literary_score

    # Log simplifié : focus sur le score minimum requis
    logger.info(T.translate("analyze_starting_semantic", thresh=min_score))

    # Chargement du modèle LLM
    engine.ensure_llm_loaded()
    
    for entry in tqdm(data, desc=T.translate("analyze_progress_bar"), leave=False):
        try:
            # Analyse via Qwen2.5-Instruct
            lit_score = engine.analyze_style(entry['text'])
            entry['literary_score'] = lit_score
            
            if lit_score >= min_score:
                final_selection.append(entry)
                logger.debug(T.translate("analyze_segment_validated", score=lit_score, file=entry['filename']))
            else:
                logger.debug(T.translate("analyze_segment_rejected", score=lit_score, thresh=min_score, file=entry['filename']))
                    
        except Exception as e:
            logger.error(T.translate("analyze_error_segment", file=entry.get('filename'), error=str(e)))
            logger.debug(traceback.format_exc())

    # Sauvegarde des métadonnées finales dans le répertoire temporaire
    final_filename = settings_internals.files.final_metadata
    output_dir = os.path.dirname(metadata_file)
    output_path = os.path.join(output_dir, final_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_selection, f, ensure_ascii=False, indent=4)
        # Affichage du nom du fichier uniquement en vert [g]
        logger.info(T.translate("analyze_final_file_generated", path=final_filename))
    except Exception as e:
        logger.error(T.translate("analyze_save_error", error=str(e)))

    # Résumé final harmonisé avec kept/total
    logger.info(T.translate("analyze_summary", kept=len(final_selection), total=len(data)))
    
    return len(final_selection)