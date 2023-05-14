import torch
import numpy as np
from pytorch_lightning import seed_everything  
import os
from scipy.io.wavfile import write
from moviepy.editor import ImageSequenceClip, concatenate_audioclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
import ffmpeg
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import openai
from multiprocessing import Pool

open_ai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = open_ai_key

from stable_diffusion.model import generate, DDIMSampler, LatentDiffusion
from config import config

from bark.bark.generation import generate_text_semantic, SAMPLE_RATE, clean_models, _clear_cuda_cache
from bark.bark.api import semantic_to_waveform

from prompts import sentence_prompt, story_prompt, story_clean_prompt

class GPTales:
    def __init__(self, story_number: int = None, speaker: str = None):
        self.story_number = story_number
        self.sample_rate = SAMPLE_RATE
        self.speaker = speaker
        self.scenes = []
        self.paragraphs = []

        if story_number is None:
            self.story_number = max([int(i.split("_")[1].split(".")[0]) for i in
                                        os.listdir("videos")]) + 1
        else:
            self.story_number = story_number

    def _gpt_call(self, prompt: str, max_tokens: int,
                  model: str = "gpt-3.5-turbo") -> str:
        for idx in range(10):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[ {"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response["choices"][0]["message"]["content"]
            except openai.error.RateLimitError:
                print(f"Error in GPT-3 call, trying again... ({idx})")
    
    def _generate_paragraph(self, story="", option=None):
        prompt = story_prompt(topics=self.topics, story=story, option=option)
        completion = self._gpt_call(prompt=prompt, max_tokens=2000)
        return completion

    def generate_story(self):
        """Generates a story with certain topics using GPT-3.5-turbo."""
        print("\nWelcome to GPTales! I am a combination of GPT-3, Bark and stable" \
              " diffusion. Together, we will create a story that is animated" \
              " and has vocals. Let's start with the story.")

        self.topics = input("What topics should the story start with? Please" \
                            " separate them with a comma. Press enter to confirm" \
                            " your choices. \n")
        print("\n")
        self.topics = self.topics.split(",")
        self.topics = [topic.strip().lower() for topic in self.topics]

        while True:

            print("Generating a new paragraph...\n")

            if len(self.paragraphs) == 0:
                new_paragraph = self._generate_paragraph()
                self.paragraphs.append(new_paragraph)
                for i in self.paragraphs: print(i, end="\n")
                option = input("How do you want to continue the story? [1/2/stop] ")
                print("\n")

            else:

                new_paragraph = self._generate_paragraph(
                    story="".join([i+"\n" for i in self.paragraphs]),
                    option=option
                )
                self.paragraphs.append(new_paragraph)
                for i in self.paragraphs: print(i, end="\n")
                option = input("How do you want to continue the story? [1/2/stop] ")
                if option == "stop": break
                print("\n")

        # To ensure that the sentences are properly split into a list
        story = [i.split("\n\n")[0] for i in self.paragraphs]
        story = [i.replace('"' ,'') for i in story]
        story = [sent_tokenize(i) for i in story]
        story = [i for j in story for i in j]
        if "The end." in story: story.remove("The end.")
        self.story = story
        
    def generate_clean_story(self):
        """Transforms the story by removing names of characters and places,
           and replacing them with descriptions. This is done to ensure that
           the story can be animated, because stable diffusion won't recognize
           such names."""
        prompt = story_clean_prompt("".join(self.story))
        names_descriptions = self._gpt_call(prompt=prompt, max_tokens=1000)
        names_descriptions = names_descriptions.split("\n")
        names_descriptions = [i.lower().replace('"', "") for i in names_descriptions]
        names_descriptions = [i.split(":") for i in names_descriptions]
        names_descriptions = [i for i in names_descriptions if len(i) > 1]
        names_descriptions = {i[0].strip(): i[1].strip() for i in names_descriptions}
        names_descriptions = {i: "["+j+"]" for i, j in names_descriptions.items()}

        story_clean = []
        for sentence in self.story:
            for name, description in names_descriptions.items():
                    sentence = sentence.lower().replace(name, description)
            story_clean.append(sentence)
        
        self.story_clean = story_clean

    def story2scene(self):
        """Uses a LLM to generate a prompt for each sentence in the story. The
           prompt can be used by the image generating model to ensure that the
           images are related to the story and are of higher quality than just
           using the sentence themselves as a prompt."""
           
        # TODO: make a postprompt using the story and GPT that is common for all
        # scenes and sets the tone
        
        # TODO: ask the LLM to generate a prompt for when music should be added
        # to the story and generate this music using BARK
        
        # TODO: make this into a sliding function to better take into account
        # the context of the story when generating the scenes
        
        # TODO: generate an image for a few sentences, intead of just one

        print("Generating scenes...")
        
        # Make a sliding window of 3 elements that is as long as the original list
        self.story_clean_sliding = [self.story_clean[i:i+3] for i in range(len(self.story_clean)-4)]
        self.story_clean_sliding.insert(0, self.story_clean[:2])
        self.story_clean_sliding.insert(0, self.story_clean[:1])
        self.story_clean_sliding.append(self.story_clean[-2:])
        self.story_clean_sliding.append(self.story_clean[-1:])
        self.story_clean_sliding = ["".join(i) for i in self.story_clean_sliding]

        # Make a nested list of 5 elements each and a remainder
        story_nested = [self.story_clean_sliding[i:i+5] for i in range(0, len(self.story_clean_sliding), 5)]

        def get_scenes(story_subset):
            prompt = sentence_prompt(story_subset)
            # Loop at least once
            while True:
                scenes = self._gpt_call(prompt=prompt, max_tokens=1000)
                scenes = scenes.split("\n")
                scenes = [i[11:].replace(".", "").strip() for i in scenes]
                if len(scenes) == len(story_subset): return scenes
        
        with Pool(10): scenes = list(tqdm([get_scenes(i) for i in story_nested], total=len(story_nested)))
        self.scenes = [i for j in scenes for i in j]
    
    def story2audio(self):
        """Generates audio for each sentence in the story."""
        # TODO: capitalization can be used for emphasis.
        for i in os.listdir("tmp/audio"): os.remove(os.path.join("tmp/audio/", i))
        before_silence = np.zeros(int(np.random.uniform(0, 0.5) * self.sample_rate))
        after_silence = np.zeros(int(np.random.uniform(0, 0.5) * self.sample_rate))
        pbar = tqdm(enumerate(self.story), leave=False, desc="Generating audio",
                    total=len(self.story))
        for idx, sentence in pbar:
            semantic_tokens = generate_text_semantic(
                sentence, history_prompt=self.speaker, min_eos_p=0.05
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

    def scene2image(self, steps: int = 20, sd_post_prompt: str = ""):
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
        pbar = tqdm(enumerate(self.scenes), leave=False, desc="Generating images",
                    total=len(self.scenes))
        for idx, scene in pbar:
            scene = scene.replace("[Prompt]", "")
            generate(prompt=scene+sd_post_prompt, model=model,
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

        # TODO: add a some noise to when the images shift
        
        # Get duration of the scene
        audio_duration = [ffmpeg.probe("tmp/audio/" + i)["format"]["duration"]
                          for i in audio_files]
        audio_duration = [float(i) for i in audio_duration]
        image_clip = ImageSequenceClip(image_filenames, durations=audio_duration)
    
        # Combine audio files
        audio_clips = [AudioFileClip("tmp/audio/" + i) for i in audio_files]    
        audio_clip = concatenate_audioclips(audio_clips)
    
        # Add audio to the video and save
        image_clip = image_clip.set_audio(audio_clip)
        # Find the last story number in the videos folder
        image_clip.write_videofile(f"videos/video_{self.story_number}.mp4", fps=24)
        
        # Remove audio and images
        for i in os.listdir("tmp/audio"): os.remove(os.path.join("tmp/audio/", i))
        for i in os.listdir("tmp/images"): os.remove(os.path.join("tmp/images/", i))


def main():
    # TODO: add voice modifiers such as [dramatic] or [music] to the story
    storyteller = GPTales(speaker="en_speaker_4")
    storyteller.generate_story()
    storyteller.generate_clean_story()
    storyteller.story2scene()
    storyteller.story2audio()
    storyteller.scene2image(steps=50, sd_post_prompt=", anime, digital art, vivid, highly detailed, 4K")
    storyteller.image2video()


if __name__=="__main__": main()