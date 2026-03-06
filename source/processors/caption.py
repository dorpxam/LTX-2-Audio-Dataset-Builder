import os
import json
import shutil
from tqdm import tqdm

from source.utils.logger import get_logger
from source.utils.translator import T

logger = get_logger("Caption")

def process(metadata_file: str, output_dir: str, config, settings_internals):
    """
    Step 4: Génération du dataset final (Audio + JSONL).
    """
    final_audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(final_audio_dir, exist_ok=True)
    
    # Récupération du nom du JSONL via settings.yaml
    jsonl_filename = settings_internals.files.dataset_jsonl
    jsonl_path = os.path.abspath(os.path.join(output_dir, jsonl_filename))

    if not os.path.exists(metadata_file):
        logger.error(T.translate("caption_metadata_not_found", path=metadata_file))
        return 0

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(T.translate("caption_json_read_error", error=str(e)))
        return 0

    if not data:
        logger.warning(T.translate("caption_no_validated_segments"))
        return 0

    # Début de l'export final
    logger.info(T.translate("caption_gen_dataset_start", dir=output_dir))
    logger.info(T.translate("caption_count_export", count=len(data)))
    
    exported_count = 0
    
    try:
        with open(jsonl_path, 'w', encoding='utf-8') as f_out:
            for entry in tqdm(data, desc=T.translate("caption_progress_bar"), leave=False):
                old_path = entry['abs_path']
                filename = os.path.basename(old_path)
                new_abs_path = os.path.abspath(os.path.join(final_audio_dir, filename))

                # Copie physique des fichiers de temp/ vers le dossier final
                if os.path.exists(old_path):
                    shutil.copy2(old_path, new_abs_path)
                else:
                    logger.warning(T.translate("caption_source_missing", path=old_path))
                    continue

                # Nettoyage pour le format JSONL
                text_clean = entry['text'].replace('"', "'") 
                
                try:
                    caption = config.template.format(
                        gender=config.gender,
                        language=config.language,
                        text=text_clean
                    )
                except Exception:
                    # Fallback si le template YAML est corrompu
                    caption = f"{config.gender}, {config.language}: {text_clean}"

                line = {
                    "audio_path": new_abs_path,
                    "caption": caption
                }

                f_out.write(json.dumps(line, ensure_ascii=False) + '\n')
                exported_count += 1

        # Logs de fin de pipeline
        logger.info(T.translate("caption_jsonl_success", path=jsonl_path))
        # Utilisation de la clé d'export finale (Count en cyan, Dir en vert)
        logger.info(T.translate("caption_export_summary", count=exported_count, dir=final_audio_dir))
        
    except Exception as e:
        logger.error(T.translate("caption_export_error", error=str(e)))
        return exported_count

    return exported_count