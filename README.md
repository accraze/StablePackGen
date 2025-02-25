# StablePackGen

StablePackGen is an AI-powered audio sample pack generator that creates complete, ready-to-use sample libraries using Stability AI's Stable Audio Open model.

## Features

- **Complete Sample Packs**: Generate entire collections of audio samples organized by category
- **Precise Duration Control**: Automatically trim samples to exact lengths for perfect timing in your productions
- **Category Organization**: Samples are sorted into logical categories (kicks, snares, hats, percussion, synths, pads, FX)
- **High-Quality Audio**: Leverages Stable Audio Open for professional-quality sound design
- **LLM-Guided Generation**: Uses Claude to help craft detailed prompts for each sample

## Requirements

- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/StablePackGen.git
cd StablePackGen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Usage

To generate a complete sample pack:

```bash
python stablepackgen.py
```

By default, this will create a techno sample pack with the following categories:
- Kicks
- Snares
- Hi-hats
- Percussion
- Synths
- Pads
- FX

All samples will be saved in their respective category folders in the current directory.

## Customization

### Modifying Sample Types

To customize the types of samples generated, edit the `generate_sample_pack_plan()` function in the script. You can:

- Add new sample categories
- Modify existing prompts
- Change duration settings
- Customize output file paths

### Changing Genres

The system is currently set up for techno, but you can easily modify the prompts for other genres like:
- House
- Ambient
- Trap
- Lo-fi
- Drum & Bass

### Generation Settings

You can adjust the generation settings in the `agent_step()` method:
- `cfg_scale`: Controls how closely the generation follows the prompt (higher = more faithful)
- `steps`: Number of diffusion steps (higher = better quality but slower)

## How It Works

StablePackGen uses a combination of:

1. **LangGraph Workflow**: Orchestrates the sample generation process
2. **Claude AI**: Helps craft detailed prompts for each sample type
3. **Stable Audio Open**: Generates high-quality audio based on text prompts
4. **Audio Processing**: Processes and trims samples to exact durations

## Requirements.txt

```
torch
torchaudio
einops
langgraph
langchain-core
langchain-anthropic
anthropic
stable-audio-tools
librosa
soundfile
numpy
typing-extensions
```

## License

MIT

## Acknowledgments

- [Stability AI](https://stability.ai/) for the Stable Audio Open model
- [Anthropic](https://www.anthropic.com/) for the Claude AI model
- [LangChain](https://www.langchain.com/) for the LangGraph workflow framework