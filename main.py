import torch
import numpy as np
from pytorch_lightning import seed_everything  
import os
from scipy.io.wavfile import write
from moviepy.editor import ImageSequenceClip, concatenate_audioclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
import ffmpeg
from tqdm import tqdm
from noisereduce import reduce_noise
from typing import List
from nltk.tokenize import sent_tokenize
import openai

open_ai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = open_ai_key

from stable_diffusion.model import generate, DDIMSampler, LatentDiffusion
from config import config

from bark.bark.generation import generate_text_semantic, SAMPLE_RATE, clean_models, _clear_cuda_cache
from bark.bark.api import semantic_to_waveform

from prompts import sentence_prompt, story_prompt

class GPTales:
    def __init__(self, story_number: int):
        self.story_number = story_number
        self.sample_rate = SAMPLE_RATE
        self.speaker = "v2/en_speaker_9"   # Female voice
        self.scenes = []
        self.paragraphs = []

    def _gpt_call(self, prompt: str, max_tokens: int,
                  model: str = "gpt-3.5-turbo") -> str:
        response = openai.ChatCompletion.create(
          model=model,
          messages=[ {"role": "user", "content": prompt}],
          max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]
    
    def _generate_paragraph(self, story=""):
        prompt = story_prompt(topics=self.topics, story=story)
        completion = self._gpt_call(prompt=prompt, max_tokens=2000)
        return completion

    def generate_story(self):
        """Generates a story with certain topics using GPT-3.5-turbo."""
        print("-"*80)
        print("Welcome to GPTales! I am a combination of GPT-3, Bark and stable" \
              " diffusion. Together, we will create a story that is animated" \
              " and has vocals. Let's start with the story.")

        self.topics = input("What topics should the story start with? Please" \
                            " separate them with a comma. Press enter to confirm" \
                            " your choices. \n")
        print("\n")
        self.topics = self.topics.split(",")
        self.topics = [topic.strip().lower() for topic in self.topics]
        
        first_paragraph = self._generate_paragraph()
        self.paragraphs.append(first_paragraph)

        while True:
            print("Generating a new paragraph...", end="\r")
            new_paragraph = self._generate_paragraph(story="".join([i+"\n" for i in self.paragraphs]))
            self.paragraphs.append(new_paragraph)
            for i in self.paragraphs: print(i, "\n")
            next = input("Do you want to continue the story? [y/n] \n")
            if next == "n": break
        
        # To ensure that the sentences are properly split into a list
        story = [sent_tokenize(i) for i in self.paragraphs]
        story = [i for j in story for i in j]
        if "The end." in story: story.remove("The end.")
        self.story = story

    def story2scene(self):
        """Uses a LLM to generate a prompt for each sentence in the story. The
           prompt can be used by the image generating model to ensure that the
           images are related to the story and are of higher quality than just
           using the sentence themselves as a prompt."""
           
        # TODO: make a postprompt using the story and GPT that is common for all
        # scenes and sets the tone
        
        # TODO: ask the LLM to generate a prompt for when music should be added
        # to the story and generate this music using BARK
    
        # Make story into a nested list of 5 elements each and a remainder
        story_nested = [self.story[i:i+5] for i in range(0, len(self.story), 5)]
        
        pbar = tqdm(story_nested, leave=False, desc="Generating scenes",
                    total=len(story_nested))
        for story_subset in pbar:
            prompt = sentence_prompt(story_subset)
            scenes = self._gpt_call(prompt=prompt, max_tokens=1000)
            scenes = scenes.split("\n")
            scenes = [i[11:].replace(".", "").strip() for i in scenes]
            assert len(scenes) == len(story_subset)
            self.scenes.extend(scenes)
    
    def story2audio(self):
        """Generates audio for each sentence in the story."""
        for i in os.listdir("tmp/audio"): os.remove(os.path.join("tmp/audio/", i))
        before_silence = np.zeros(int(np.random.uniform(0, 0.5) * self.sample_rate))
        after_silence = np.zeros(int(np.random.uniform(0, 0.5) * self.sample_rate))
        pbar = tqdm(enumerate(self.story), leave=False, desc="Generating audio",
                    total=len(self.story))
        for idx, sentence in pbar:
            semantic_tokens = generate_text_semantic(
                sentence, history_prompt=self.speaker, temp=0.5, min_eos_p=0.05
            )
            audio_array = semantic_to_waveform(semantic_tokens,
                                               history_prompt=self.speaker)
            audio_array = np.concatenate([before_silence.copy(), audio_array,
                                          after_silence.copy()])
            audio_array = np.clip(audio_array, -0.2, audio_array.max())
            # audio_array = reduce_noise(y=audio_array, sr=self.sample_rate)
            write(f"tmp/audio/audio_{idx}.wav", self.sample_rate, audio_array)
            
        clean_models()
        _clear_cuda_cache()

    def scene2image(self, steps: int = 20):
        """Uses the stable diffusion 2 model to generate visuals for each
           scene."""
        for i in os.listdir("tmp/images"): os.remove(os.path.join("tmp/images/", i))
        seed_everything(42)
        checkpoint = torch.load("checkpoints/v2-1_512-ema-pruned.ckpt")
        model = LatentDiffusion(**config["model"]["params"])
        _ = model.load_state_dict(checkpoint["state_dict"], strict=False)
        model.cuda()
        model.eval()
        sampler = DDIMSampler(model, device="cuda")
        post_prompt = ", anime, digital art, vivid, highly detailed, 4k, 8k"
        pbar = tqdm(enumerate(self.scenes), leave=False, desc="Generating images",
                    total=len(self.scenes))
        for idx, scene in pbar:
            scene = scene.replace("[Prompt]", "")
            generate(prompt=scene+post_prompt, model=model,
                     sampler=sampler, steps=steps, path=f"tmp/images/scene_{idx}.png")
    
    def image2video(self):
        """Combines the images and audio into a video."""
        # TODO: add the text to the images
        # TODO: add music to the video
        image_filenames = ["tmp/images/" + i for i in os.listdir("tmp/images")]
        image_filenames = sorted(image_filenames,
                                 key=lambda x: int(x.split("_")[1].split(".")[0]))
        audio_files = os.listdir("tmp/audio")
        audio_files = sorted(audio_files,
                             key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        # Get duration of the scene
        audio_duration = [ffmpeg.probe("tmp/audio/" + i)["format"]["duration"] for i
                          in audio_files]
        audio_duration = [float(i) for i in audio_duration]
        image_clip = ImageSequenceClip(image_filenames, durations=audio_duration)
    
        # Combine audio files
        audio_clips = [AudioFileClip("tmp/audio/" + i) for i in audio_files]    
        audio_clip = concatenate_audioclips(audio_clips)
    
        # Add audio to the video and save
        image_clip = image_clip.set_audio(audio_clip)
        image_clip.write_videofile(f"videos/video_{self.story_number}.mp4", fps=24)
        
        # Remove audio and images
        for i in os.listdir("tmp/audio"): os.remove(os.path.join("tmp/audio/", i))
        for i in os.listdir("tmp/images"): os.remove(os.path.join("tmp/images/", i))


def main():
    # TODO: add voice modifiers such as [dramatic] or [music] to the story
    storyteller = GPTales(story_number=3)
    storyteller.generate_story()
    storyteller.story2scene()
    storyteller.story2audio()
    storyteller.scene2image()
    storyteller.image2video()


if __name__=="__main__": main()