import torch
import numpy as np
from pytorch_lightning import seed_everything  
import os
from scipy.io.wavfile import write
from moviepy.editor import ImageSequenceClip, concatenate_audioclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
import ffmpeg
from tqdm import tqdm

from stable_diffusion.model import generate, DDIMSampler, LatentDiffusion
from config import config

from bark.bark.generation import generate_text_semantic, SAMPLE_RATE, preload_models
from bark.bark.api import semantic_to_waveform

import openai
from api_keys import open_ai_key
openai.api_key = open_ai_key
from prompts import sentence_prompt

STORY_NUMBER = 2


def gen_story(keywords):
    pass

def story2scene(story):
    """Uses a LLM to generate a prompt for each sentence in the story. The
       prompt can be used by the image generating model to ensure that the
       images are related to the story and are of higher quality than just
       using the sentence as a prompt."""
       
    # Delete previously generated scenes
    os.remove(f"scenes/scene_{STORY_NUMBER}.txt")

    # Make story into a nested list of 5 elements each and a remainder
    story = [story[i:i+5] for i in range(0, len(story), 5)]
    
    for i in tqdm(story, leave=False, desc="Generating scenes", total=len(story)):
        prompt = sentence_prompt(i)
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo", messages=[ {"role": "user", "content": prompt}],
          max_tokens=1000
        )
        completion = response["choices"][0]["message"]["content"]
        completion = completion.split("\n")
        completion = [i[11:].replace(".", "").strip() for i in completion]
        assert len(completion) == len(i)
        with open(f"scenes/scene_{STORY_NUMBER}.txt", "a") as f:
            for i in completion: f.write(i + "\n")

def story2audio(story):
    for i in os.listdir("audio"): os.remove(os.path.join("audio/", i))
    preload_models()
    GEN_TEMP = 0.6
    # SPEAKER = "v2/en_speaker_3" # Good, but slow
    SPEAKER = "v2/en_speaker_5" 
    silence = np.zeros(int(0.3 * SAMPLE_RATE))  # quarter second of silence
    for idx, sentence in tqdm(enumerate(story), leave=False, desc="Generating audio", total=len(story)):
        semantic_tokens = generate_text_semantic(sentence, history_prompt=SPEAKER,
                                                 temp=GEN_TEMP, min_eos_p=0.05)
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        audio_array = np.concatenate([silence.copy(), audio_array, silence.copy()])
        write(f"audio/audio_{idx}.wav", SAMPLE_RATE, audio_array)

def scene2image(steps=20):
    scenes = open(f"scenes/scene_{STORY_NUMBER}.txt", "r").readlines()
    scenes = [i.replace("\n", "") for i in scenes]
    for i in os.listdir("images"): os.remove(os.path.join("images/", i))
    seed_everything(42)
    checkpoint = torch.load("checkpoints/v2-1_512-ema-pruned.ckpt")
    model = LatentDiffusion(**config["model"]["params"])
    _ = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.cuda().eval()
    sampler = DDIMSampler(model, device="cuda")
    post_prompt = ", anime, vivid, highly detailed, 4k, 8k"
    for idx, scene in tqdm(enumerate(scenes), leave=False, desc="Generating images", total=len(scenes)):
        scene = scene.replace("[Prompt]", "")
        generate(prompt=scene+post_prompt, model=model,
                 sampler=sampler, steps=steps, path=f"images/scene_{idx}.png")

def image2video():
    # Get the images and audio files and sort them
    image_filenames = ["images/" + i for i in os.listdir("images")]
    image_filenames = sorted(image_filenames, key=lambda x: int(x.split("_")[1].split(".")[0]))
    audio_files = os.listdir("audio")
    audio_files = sorted(audio_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    
    # Get duration of the scene
    audio_duration = [ffmpeg.probe("audio/" + i)["format"]["duration"] for i
                      in audio_files]
    audio_duration = [float(i) for i in audio_duration]
    image_clip = ImageSequenceClip(image_filenames, durations=audio_duration)

    # Combine audio files
    audio_clips = [AudioFileClip("audio/" + i) for i in audio_files]    
    audio_clip = concatenate_audioclips(audio_clips)
    audio_clip.write_audiofile("audio.wav")

    # Add audio to the video and save
    image_clip = image_clip.set_audio(audio_clip)
    image_clip.write_videofile(f"videos/video_{STORY_NUMBER}.mp4", fps=24)


def main():
    story = open(f"stories/story_{STORY_NUMBER}.txt", "r").readlines()
    story = [i.replace("\n", "") for i in story]

    # story2scene(story)
    scene2image(steps=50)
    story2audio(story)
    image2video()


if __name__=="__main__": main()