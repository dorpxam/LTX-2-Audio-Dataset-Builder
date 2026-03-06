# LTX-2 Audio Dataset Builder

**LTX-2 Audio Dataset Builder** is a specialized tool designed to automate the creation of high-quality audio datasets. It transforms raw audio sources into clean, curated, and captioned segments specifically optimized for training the **LTX-2** model in audio-only mode.

## Introduction of the Pipeline

- **STAGE 1**: **<code style="color : YellowGreen">VAD Segmentation & Normalization</code>**: Performs intelligent slicing. Segments are immediately processed: converted to mono, resampled, and loudness-normalized to ensure acoustic consistency.
- **STAGE 2**: **<code style="color : YellowGreen">ASR & Lexical Scoring</code>**: Filters segments based on transcription redundancy. It calculates a lexical density score to eliminate repetitive or "garbage" segments often found in automated speech-to-text outputs.
- **STAGE 3**: **<code style="color : YellowGreen">LLM Qualitative Analysis</code>**: Leverages LLM models to rate the **literary richness** and **phonetic complexity** of each segment on a scale of 1 to 10. Only segments meeting the minimum qualitative threshold are preserved.
- **STAGE 4**: **<code style="color : YellowGreen">Dataset Consolidation</code>**: Finalizes the dataset by exporting the validated audio files and generating a structured `dataset.jsonl` file ready for training.

## Installation

1. Clone the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
> Note: The requirements.txt is adapted to my Window OS with CUDA 12.8 to avoid installation probelem with torchcodec. Of course, use your own python environment (or conda) to install the few dependencies of the project. 

## Usage

Run the pipeline via the command line:

```bash
python main.py "path/to/audio_source" "output_directory" --config "path/to/config.yaml"
```
> Note: A template with the default settings is available in the root of the projet : `config.yaml.template`, copy and rename to anything you want `my_dataset_1.yaml` and use it as input with your own settings. Be careful to adjust de `gender` and the `language` in the YAML file to be sure that Qwen3-ASR do not translate your caption. 

**Arguments:**

* `audio_source`: Path to the source file or directory.
* `output_dir`: Target destination for the final dataset.
* `--config`: (Optional) Path to a custom YAML configuration. (fallback on internal default `config.yaml`)
* `--lang`: (Optional) Console interface language (`en` or `fr`). This is independent of the dataset language defined in `config.yaml`.

## Configuration

The `config.yaml` file is the heart of the process, allowing you to control every stage of the pipeline:

* **hardware**: Configure CUDA device, 8-bit quantization, and Flash Attention 2.
* **segmentation**: Set VAD thresholds, duration limits (min/max), and audio export specifications (bit depth, sample rate).
* **scoring**: Define the `max_redundancy` factor to filter repetitive transcripts.
* **analyze**: Set the `min_literary_score` to ensure the dataset only contains high-quality language.
* **caption**: Define the gender, language, and the prompt template used for the final metadata.

## Good Practice

> Warning: These practical tips are based on my own experience; I do not claim to offer the ideal method, but it works very well for me, so I leave it to you to use what interests you in this information.

>Essentially, the tool is a CLI. Therefore, to work more easily with different datasets, it's best to avoid the pitfalls of handling numerous files. Ideally, organize your datasets in a dedicated directory:

```bash
├── LTX-2
│   ├── dataset_1
│   │   ├── builder
│   │   │   ├── builder.bat 
│   │   │   ├── config.yaml                                         <- your specific config.yaml
│   │   │   ├── launch.bat
│   │   │   ├── my_audio_input.flac
│   │   │   └── my_audio_input.log                                  <- created
│   │   ├── dataset
│   │   │   ├── audio                                               <- created and filled
│   │   │   │   ├── segment_0150.flac
│   │   │   │   ├── segment_0353.flac
│   │   │   │   ├── segment_0632.flac
│   │   │   │   └── ...
│   │   │   ├── config.toml
│   │   │   └── dataset.jsonl                                       <- created
│   │   ├── 01-latent-pre-caching.bat
│   │   ├── 02-text-encoder-pre-caching.bat
│   │   ├── 03-training.bat
│   │   └── launch.bat
│   ├── dataset_2
│   │   ├── builder
│   │   │   ├── ...
│   │   ├── dataset
│   │   │   ├── ...
...
```

> All others file are 'musubi-tuner' related. 

> To simplify the process, a simple `launch.bat` in subdirectory `builder` isolate the 3 parameters to pass to this `main.py` of the toolkit.

```bash
echo off

set audio_input_path="path/to/your/input_audio_file"
set output_directory="path/to/your/output_directory"
set config_yaml_path="path/to/your/config.yaml"

builder.bat %audio_input_path% %output_directory% %config_yaml_path%
```

> `launch.bat` call `builder.bat` that is a generic bash file parametrized for your installation :

```bash
echo off

set installation_dir="path/to/your/installation/directory"
set conda_env="your/conda/env/name"

set arg1=%1
set arg2=%2
set arg3=%3

call C:\anaconda3\condabin\activate.bat
c:
cd %installation_dir%
call conda activate %conda_env%

python main.py %arg1% %arg2% --config %arg3%
pause
```

> Of course, here my installation is on my `C:` drive while my datasets are on my `F:` drive. So the `c:` is just a windows-like way to switch on `C:` drive required before the `cd` (change directory).

> The console will be automatically cleared before the start of the process. And you will see something like that (windows terminal) :

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](/assets/screenshot.png)

## Acknowledgments

A special shoutout to **@AkaneTendo25** for his remarkable work on the [musubi-tuner](https://github.com/AkaneTendo25/musubi-tuner) fork, specifically tailored for LTX-2 training.