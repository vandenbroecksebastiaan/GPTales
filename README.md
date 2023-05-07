# GPTales

Presenting GPTales, a python script that generates short stories for children
using GPT-3, a powerful language generation model from OpenAI. Before generating
the story, it asks for some topics that the user is interested in. Afterwards,
stable diffusion is used to create visualizations that bring the story to life,
and Bark, a model that generates a voice, reads the story aloud with exceptional
clarity and realism. Finally, everything is combined in a video. The culmination
of these groundbreaking technologies is a mesmerizing video that is sure to
delight and inspire children of all ages.

## Usage

You need about 12GB of VRAM and there should be an OpenAI API key in your environment:
```bash
export OPENAI_API_KEY=YOUR_API_KEY_HERE
```

After running main.py, GPTales will ask you for some topics to generate a story
you need to provide at least one. Next, it will generate paragraphs one by one
and you can choose yourself whether the story continues or not.

## Example

Examples can be found in the videos folder.

<video src="video_6.mp4" width=200/>

## License
The code under stable_diffusion/ has been adapted from and provided under MIT
license by [Stability-AI](https://github.com/Stability-AI). Bark, created by
Suno AI, has been provided under the MIT license as well.
