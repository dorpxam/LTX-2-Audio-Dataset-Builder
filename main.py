import os
import time
import argparse
import shutil
import logging

from source.utils.warning import silence_warnings
silence_warnings()

from source.common.config import load_all_configs

from source.processors.segmentation import process as step_segmentation
from source.processors.scoring import process as step_scoring
from source.processors.analyze import process as step_analyze
from source.processors.caption import process as step_caption

from source.utils.translator import T
from source.utils.logger import setup_mxp_logging
from source.utils.console import Console
from source.__version__ import __app_name__, __version__, __author__

console = Console()
logger = logging.getLogger("MXP")

def process(pipeline_cfg):
    start_time = time.time()
    
    cfg = pipeline_cfg.config
    settings = pipeline_cfg.settings
    engine_cfg = pipeline_cfg.engine
    
    temp_dir = settings.internals.paths.temp_dir
    output_dir = pipeline_cfg.output_dir

    final_audio_path = os.path.join(pipeline_cfg.output_dir, "audio")
    if os.path.exists(final_audio_path):
        shutil.rmtree(final_audio_path)
    
    console.clear()
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    meta_scored = os.path.join(temp_dir, settings.internals.files.scored_metadata)
    meta_final = os.path.join(temp_dir, settings.internals.files.final_metadata)

    console.separator()
    console.print(f"[x]{__app_name__} | VERSION {__version__} | BY {__author__}")
    console.separator()

    try:
        logger.info(T.translate("main_pipeline_start"))
        console.separator()
        logger.info(T.translate("main_output_directory", path=output_dir))
        console.separator()

        logger.info(T.translate("main_step_segmentation"))
        console.separator()
        step_segmentation(pipeline_cfg.audio_source, temp_dir, cfg.segmentation, settings.models.vad)
        console.separator()

        logger.info(T.translate("main_step_scoring"))
        console.separator()
        count_scored = step_scoring(
            temp_dir, 
            cfg.scoring, 
            engine_cfg, 
            cfg.segmentation, 
            settings.internals, 
            cfg.caption.language
        )
        if count_scored == 0:
            logger.error(T.translate("main_error_no_segments_scoring"))
            return
        console.separator()

        logger.info(T.translate("main_step_analyze"))
        console.separator()
        count_final = step_analyze(meta_scored, cfg.analyze, engine_cfg, settings.internals)
        if count_final == 0:
            logger.error(T.translate("main_error_no_segments_analyze"))
            return
        console.separator()

        logger.info(T.translate("main_step_caption"))
        console.separator()
        step_caption(meta_final, output_dir, cfg.caption, settings.internals)
        console.separator()

        duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        logger.info(T.translate("main_pipeline_complete"))
        console.separator()
        logger.info(T.translate("main_total_duration", duration=duration))
        console.separator()

    except Exception as e:
        logger.error(T.translate("main_error_pipeline", error=str(e)), exc_info=True)
    
    finally:
        if os.path.exists(temp_dir):
            logger.info(T.translate("main_cleaning_temp"))
            shutil.rmtree(temp_dir)

def run_stats(pipeline_cfg):
    final_audio_path = os.path.join(pipeline_cfg.output_dir, "audio")
    if os.path.isdir(final_audio_path):
        from source.utils.stats import generate_post_pipeline_stats
        exts = pipeline_cfg.settings.internals.audio.supported_extensions
        generate_post_pipeline_stats(final_audio_path, exts, console, logger)
        console.separator()

def main():
    parser = argparse.ArgumentParser(description=__app_name__)
    parser.add_argument("audio_source", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--config", type=str, default=None, help="Path to a custom config.yaml")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "fr"])
    args = parser.parse_args()

    T.set_language(args.lang)
    setup_mxp_logging(audio_path=args.audio_source)

    pipeline_cfg = load_all_configs(
        audio_source=args.audio_source,
        output_dir=args.output_dir,
        user_config_path=args.config
    )

    process(pipeline_cfg)
    run_stats(pipeline_cfg)

if __name__ == "__main__":
    main()