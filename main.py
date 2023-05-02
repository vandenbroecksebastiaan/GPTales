import torch
import numpy as np
from pytorch_lightning import seed_everything  
from scipy.io.wavfile import write, read
import os
from moviepy.editor import ImageSequenceClip, concatenate_audioclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
import ffmpeg
from tqdm import tqdm

from stable_diffusion.model import generate, DDIMSampler, LatentDiffusion
from config import config

from bark.bark.generation import generate_text_semantic, SAMPLE_RATE, preload_models
from bark.bark.api import semantic_to_waveform

print("SAMPLE_RATE", SAMPLE_RATE)

def story2audio(story):
    for i in os.listdir("audio"): os.remove(os.path.join("audio/", i))
    preload_models()
    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_3"
    silence = np.zeros(int(0.3 * SAMPLE_RATE))  # quarter second of silence
    for idx, sentence in enumerate(story):
        semantic_tokens = generate_text_semantic(sentence, history_prompt=SPEAKER,
                                                 temp=GEN_TEMP, min_eos_p=0.05)
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        audio_array = np.concatenate([silence.copy(), audio_array, silence.copy()])
        write(f"audio/audio_{idx}.wav", SAMPLE_RATE, audio_array)

def scene2image(scenes):
    for i in os.listdir("images"): os.remove(os.path.join("images/", i))
    seed_everything(42)
    checkpoint = torch.load("checkpoints/v2-1_512-ema-pruned.ckpt")
    model = LatentDiffusion(**config["model"]["params"])
    _ = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.cuda().eval()
    sampler = DDIMSampler(model, device="cuda")
    post_prompt = ", anime, paint sky, highly detailed, 4k, 8k"
    for idx, scene in tqdm(enumerate(scenes), total=len(scenes)):
        generate(prompt=scene+post_prompt, model=model,
                 sampler=sampler, steps=40, path=f"images/scene_{idx}.png")

def image2video(video_name):
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
    image_clip.write_videofile(f"videos/video_{video_name}.mp4", fps=24)

def main():
    story = open("story.txt", "r").readlines()
    story = [i.replace("\n", "") for i in story]
    scenes = open("scene.txt", "r").readlines()
    scenes = [i.replace("\n", "") for i in scenes]

    scene2image(scenes)
    story2audio(story)
    image2video(video_name="0")



if __name__=="__main__": main()