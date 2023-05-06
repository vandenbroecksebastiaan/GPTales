# GPTales

Presenting GPTales, a python script that generates short stories for children
using GPT-3, a powerful language generation model from OpenAI. Before generating
the story, it asks for some topics that the user is interested in. Afterwards,
stable diffusion is used to create visualizations of the story, and Bark, a
model that generates a voice, reads the story aloud. Finally, everything is
combined in a video.

## Usage

You need to have an OpenAI API key in your environment:
```bash
export OPENAI_API_KEY=YOUR_API_KEY_HERE
```

## Example

Examples can be found in the videos folder.

## License
The code under stable_diffusion/ has been adapted from and provided under MIT
license by [Stability-AI](https://github.com/Stability-AI). Bark, created by
Suno AI, has been provided under the MIT license as well.
