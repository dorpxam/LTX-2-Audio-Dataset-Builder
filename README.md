# LTX-2 Audio Dataset Builder

**LTX-2 Audio Dataset Builder** is a specialized tool designed to automate the creation of high-quality audio datasets. It transforms raw audio sources into clean, curated, and captioned segments specifically optimized for training the **LTX-2** model in audio-only mode.

## The 4-Step Pipeline

1. **VAD Segmentation & Normalization**: Performs intelligent slicing using **Silero VAD**. Segments are immediately processed: converted to mono, resampled (48kHz), and loudness-normalized (-16.0 LUFS) to ensure acoustic consistency.
2. **Lexical Scoring**: Filters segments based on transcription redundancy. It calculates a lexical density score to eliminate repetitive or "garbage" segments often found in automated speech-to-text outputs.
3. **Qualitative Analysis**: Leverages LLM models (**Qwen**) to rate the **literary richness** and **phonetic complexity** of each segment on a scale of 1 to 10. Only segments meeting the minimum qualitative threshold are preserved.
4. **Dataset Consolidation**: Finalizes the dataset by exporting the validated audio files and generating a structured `dataset.jsonl` file ready for training.

## Installation

This project requires **Python 3.12**. For Windows users with NVIDIA GPUs:

1. Clone the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
*Note: Torch versions are pinned to ensure stability with CUDA 12.8 and torchcodec on Windows.*

## Configuration (`config.yaml`)

The `config.yaml` file is the heart of the process, allowing you to control every stage of the pipeline:

* **hardware**: Configure CUDA device, 8-bit quantization, and Flash Attention 2.
* **segmentation**: Set VAD thresholds, duration limits (min/max), and audio export specifications (bit depth, sample rate).
* **scoring**: Define the `max_redundancy` factor to filter repetitive transcripts.
* **analyze**: Set the `min_literary_score` to ensure the dataset only contains high-quality language.
* **caption**: Define the gender, language, and the prompt template used for the final metadata.

## Usage

Run the pipeline via the command line:

```bash
python main.py "path/to/source" "output_directory" --config "path/to/config.yaml"
```

**Key Arguments:**

* `audio_source`: Path to the source file or directory.
* `output_dir`: Target destination for the final dataset.
* `--lang`: **Console interface language** (`en` or `fr`). This is independent of the dataset language defined in `config.yaml`.
* `--config`: (Optional) Path to a custom YAML configuration.

## Acknowledgments

A special shoutout to **@AkaneTendo25** for his remarkable work on the [musubi-tuner](https://github.com/AkaneTendo25/musubi-tuner) fork, specifically tailored for LTX-2 training.