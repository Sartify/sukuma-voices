<p align="center">
  <img src="https://github.com/Sartify/sukuma-voices/raw/main/scripts/sukuma.png" alt="Sukuma Voices" width="800">
</p>


# Sukuma Voices 🎙️

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset19Hours-yellow)](https://huggingface.co/datasets/sartifyllc/Sukuma-Voices)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-DatasetACL-yellow)](https://huggingface.co/datasets/sartifyllc/Sukuma-Voices-ACL)
[![HuggingFace TTS](https://img.shields.io/badge/🤗%20HuggingFace-ModelTTS-red)](https://huggingface.co/sartifyllc/sukuma-voices-tts)
[![HuggingFace TTS](https://img.shields.io/badge/🤗%20HuggingFace-ModelASR-gree)](https://huggingface.co/sartifyllc/sukuma-voices-asr)
[![Paper](https://img.shields.io/badge/Paper-ACL%202025-blue)]()

**The first publicly available speech corpus for Sukuma (Kisukuma)**, a Bantu language spoken by approximately 10 million people in northern Tanzania.

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 6,871 |
| Total Duration | 19.56 hours |
| Average Duration | 10.25 ± 4.15 seconds |
| Duration Range | 1.40 - 30.36 seconds |
| Total Words | 140,325 |
| Unique Vocabulary | 21,366 |
| Average Words/Sample | 20.4 |
| Speaking Rate | 121.6 WPM |

## 🎯 Supported Tasks

- **Automatic Speech Recognition (ASR)** — Converting Sukuma speech to text
- **Text-to-Speech (TTS)** — Synthesizing natural-sounding Sukuma speech
- **Cross-lingual Speech Processing** — Research between Swahili and Sukuma

## 🚀 Quick Start

### Installation

```bash
pip install datasets transformers librosa
```

### Load the Dataset

```python
from datasets import load_dataset

# Load the dataset from HuggingFace
dataset = load_dataset("sartifyllc/sukuma-voices")

# Access train and test splits
train_data = dataset["train"]
test_data = dataset["test"]

# View a sample
print(train_data[0])
```

### ASR Inference Example

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained("sartifyllc/sukuma-voices-asr")
processor = WhisperProcessor.from_pretrained("sartifyllc/sukuma-voices-asr")

# Load and preprocess audio
audio_array = ...  # Your audio as numpy array at 16kHz

input_features = processor(
    audio_array, 
    sampling_rate=16000, 
    return_tensors="pt"
).input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# Decode
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

## 📁 Repository Structure

```
sukuma-voices/
├── README.md
├── LICENSE
├── scripts/
    ├── train_asr.py
    └── train_tts.py
```

## 📈 Baseline Results

### ASR Performance (Whisper Large V3)

| Metric | Original Speech | Synthetic Speech |
|--------|-----------------|------------------|
| Final WER | 25.19% | 32.60% |
| Min WER | 22.01% | 29.97% |
| WER Reduction | 82.94% | 78.93% |

### TTS Performance (Orpheus 3B v0.1)

| Metric | Score |
|--------|-------|
| Mean Opinion Score (MOS) | 3.9 ± 0.15 |
| Human Recording MOS | 4.6 ± 0.1 |

## 📝 Example Data

| Language | Text |
|----------|------|
| **Sukuma** | Umunhu ngwunuyo agabhalelaga chiza abhanhu bhakwe, kunguyo ya kikalile kakwe akagubhatogwa na gubhambilija abho bhali mumakoye. |
| **English** | This person raises his people well, because of his good behavior, of loving people and helping his colleagues who are in trouble, in their lives. |

## 🔬 Reproducing Results

### Training ASR

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
python train_asr.py
```

### Training TTS

```bash
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
python scripts/train_tts.py 
```

## ⚠️ Known Limitations

- **Domain Specificity**: Data sourced from biblical texts may not fully represent everyday conversational Sukuma
- **Diacritic Variations**: Sukuma has two written forms; this dataset uses the non-diacritic version
- **Speaker Diversity**: Limited speaker diversity from a single recording source

## 📜 Citation

If you use this dataset in your research, please cite:

```bibtex
@inproceedings{mgonzo2025sukuma,
  title={Learning from Scarcity: Building and Benchmarking Speech Technology for Sukuma},
  author={Mgonzo, Macton and Oketch, Kezia and Etori, Naome and Mang'eni, Winnie and Nyaki, Elizabeth and Mollel, Michael S.},
  booktitle={Proceedings of the Association for Computational Linguistics},
  year={2025},
  institution={Brown University, University of Notre Dame, University of Minnesota - Twin Cities, Pawa AI, Sartify Company Limited}
}
```

## 👥 Authors

| Name | Affiliation | Contact |
|------|-------------|---------|
| **Macton Mgonzo** | Brown University | macton_mgonzo@brown.edu |
| **Kezia Oketch** | University of Notre Dame | |
| **Naome Etori** | University of Minnesota - Twin Cities | |
| **Winnie Mang'eni** | Pawa AI | |
| **Elizabeth Nyaki** | Pawa AI, Sartify Company Limited | |
| **Michael S. Mollel** | Pawa AI, Sartify Company Limited |michael.mollel@sartify.com |

## 🙏 Acknowledgments

We would like to express our gratitude to [Sartify Company Limited](https://www.sartify.com/) and [Pawa AI](https://www.pawa-ai.com/) for their instrumental role in initiating this project and for providing the data access necessary to develop and evaluate our models. We also extend our sincere thanks to all the volunteers who generously dedicated their time to the evaluation process.

## 📄 License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## 🤝 Contributing

We welcome contributions to expand and improve this dataset! Areas of interest include:

- Additional Sukuma speech data beyond religious content
- Conversational and everyday language recordings
- Multi-speaker recordings
- Diacritic-annotated transcriptions

Please open an issue or submit a pull request to contribute.

## 📧 Contact

For questions, collaborations, or feedback, please:
- Open an issue on this repository
- Contact: macton_mgonzo@brown.edu, info@sartify.com

---

<p align="center">
  <i>This dataset represents an important step toward inclusive speech technology for African languages.</i>
</p>

<p align="center">
  <a href="https://www.sartify.com/">Sartify</a> •
  <a href="https://www.pawa-ai.com/">Pawa AI</a> •
  <a href="https://huggingface.co/datasets/your-username/sukuma-voices">HuggingFace</a>
</p>
