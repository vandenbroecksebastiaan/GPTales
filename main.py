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
from api_keys import open_ai_key
openai.api_key = open_ai_key

from stable_diffusion.model import generate, DDIMSampler, LatentDiffusion
from config import config

from bark.bark.generation import generate_text_semantic, SAMPLE_RATE
from bark.bark.api import semantic_to_waveform

from prompts import sentence_prompt, story_prompt


class GPTales:
    def __init__(self, story_number: int):
        self.story_number = story_number
        self.sample_rate = SAMPLE_RATE
        self.speaker = "v2/en_speaker_9"   # Female voice

        self.scenes = []

    def _gpt_call(self, prompt: str, max_tokens: int,
                  model: str = "gpt-3.5-turbo") -> str:
        response = openai.ChatCompletion.create(
          model=model,
          messages=[ {"role": "user", "content": prompt}],
          max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]

    def generate_story(self, topics: List[str]):
        """Generates a story with certain topics using GPT-3.5-turbo."""
        print("Generating story")
        prompt = story_prompt(topics)
        story = self._gpt_call(prompt=prompt, max_tokens=2000)
        # To ensure that the sentences are properly split into a list
        story = sent_tokenize(story)
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
        for i in os.listdir("audio"): os.remove(os.path.join("audio/", i))
        before_silence = np.zeros(int(np.random.uniform(0, 0.5) * self.sample_rate))
        after_silence = np.zeros(int(np.random.uniform(0, 0.5) * self.sample_rate))
        # pbar = tqdm(enumerate(self.story), leave=False, desc="Generating audio",
        #             total=len(self.story))
        pbar = tqdm(enumerate(self.story[:1]), leave=False, desc="Generating audio",
                    total=len(self.story[:1]))
        for idx, sentence in pbar:
            semantic_tokens = generate_text_semantic(
                sentence, history_prompt=self.speaker, temp=0.5, min_eos_p=0.05
            )
            audio_array = semantic_to_waveform(semantic_tokens,
                                               history_prompt=self.speaker)
            audio_array = np.concatenate([before_silence.copy(), audio_array,
                                          after_silence.copy()])
            audio_array = np.clip(audio_array, -0.2, audio_array.max())
            audio_array = reduce_noise(y=audio_array, sr=self.sample_rate)
            write(f"audio/audio_{idx}.wav", self.sample_rate, audio_array)

    def scene2image(self, steps: int = 20):
        """Uses the stable diffusion 2 model to generate visuals for each
           scene."""
        for i in os.listdir("images"): os.remove(os.path.join("images/", i))
        seed_everything(42)
        checkpoint = torch.load("checkpoints/v2-1_512-ema-pruned.ckpt")
        model = LatentDiffusion(**config["model"]["params"])
        _ = model.load_state_dict(checkpoint["state_dict"], strict=False)
        # model.cuda()
        model.eval()
        sampler = DDIMSampler(model, device="cuda")
        post_prompt = ", anime, digital art, vivid, highly detailed, 4k, 8k"
        pbar = tqdm(enumerate(self.scenes), leave=False, desc="Generating images",
                    total=len(self.scenes))
        for idx, scene in pbar:
            scene = scene.replace("[Prompt]", "")
            generate(prompt=scene+post_prompt, model=model,
                     sampler=sampler, steps=steps, path=f"images/scene_{idx}.png")
    
    def image2video(self):
        """Combines the images and audio into a video."""
        # TODO: add the text to the images
        # TODO: add music to the video
        image_filenames = ["images/" + i for i in os.listdir("images")]
        image_filenames = sorted(image_filenames,
                                 key=lambda x: int(x.split("_")[1].split(".")[0]))
        audio_files = os.listdir("audio")
        audio_files = sorted(audio_files,
                             key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        # Get duration of the scene
        audio_duration = [ffmpeg.probe("audio/" + i)["format"]["duration"] for i
                          in audio_files]
        audio_duration = [float(i) for i in audio_duration]
        image_clip = ImageSequenceClip(image_filenames, durations=audio_duration)
    
        # Combine audio files
        audio_clips = [AudioFileClip("audio/" + i) for i in audio_files]    
        audio_clip = concatenate_audioclips(audio_clips)
        # audio_clip.write_audiofile("audio.wav")
    
        # Add audio to the video and save
        image_clip = image_clip.set_audio(audio_clip)
        image_clip.write_videofile(f"videos/video_{self.story_number}.mp4", fps=24)


def main():
    # story = open(f"stories/story_{STORY_NUMBER}.txt", "r").readlines()
    # story = [i.replace("\n", "") for i in story]

    # TODO: add voice modifiers such as [dramatic] or [music] to the story
    topics = ["astronaut", "lost in space", "dog best friend"]
    storyteller = GPTales(story_number=3)
    # storyteller.generate_story(topics=topics)
    # storyteller.story2scene()
    storyteller.story = ['Once upon a time, there was an astronaut named Max who loved exploring space.', 'He had a passion for discovering new planets and galaxies.', 'One day, while he was conducting a mission to explore a new planet, his spacecraft developed a technical fault.', 'The engine malfunctioned and he got lost in space.', 'Max was now stranded on a strange planet, alone and frightened.', 'He had no idea how he would survive on this strange planet, and all he had was his spacesuit and a few supplies.', 'As the hours turned into days and the days into weeks, Max was feeling more and more homesick.', 'He missed his family and his best friend back on Earth.', 'In his loneliness, he remembered how his dog, Buddy, always stayed by his side, no matter what.', 'He wished Buddy was there to comfort him in his times of despair.', 'Suddenly, Max had an idea.', "He remembered Buddy's unerring sense of smell and cleverness, which helped Max through difficult situations in the past.", 'With new hope in his heart, Max quickly wrote a letter to Buddy, attached it to his spacecraft, and sent it off into space.', 'He hoped with all his heart that Buddy would find it and understand what he needed.', 'A few days passed, and Max was at his weakest point.', 'The only thing keeping him alive was his will and determination to survive.', 'Then, he saw something shining in the distance, and soon realized it was his spacecraft, as it had returned.', 'To his joy, a loyal Buddy was inside the spacecraft, wagging his tail.', "Buddy's keen sense of smell and intelligence had helped him find Max.", 'As soon as Max opened the hatch, Buddy jumped onto him, and the two hugged each other with joy.', 'Together, Max and Buddy made the best of their situation and worked together to create an escape plan.', 'Finally, they managed to fix the spacecraft and blasted off into space, heading back home.', 'The moment they reached Earth, Max and Buddy were surrounded by happy faces and loved ones.', 'Max realized that no matter where he was, he would always have a special bond with his faithful friend.', 'From that day on, Max vowed to never forget the importance of loyalty and friendship.', 'The two cherished each other and lived happily ever after.']
    storyteller.scenes = ['An astronaut in a spacesuit, standing on a rocky planet with two moons in the background', 'A telescope pointing towards the stars, a galaxy poster on the wall, a stack of books about space on a table', 'A spaceship hovering above a strange planet, with smoke coming out of the engine, and the astronaut looking out the window anxiously', 'A view of the galaxy from the spacecraft window, with the stars and planets passing by', 'An astronaut sitting alone on the ground, surrounded by strange alien structures and plants, with a look of desperation on his face', 'An astronaut, alone, standing on a strange planet with a few supplies and a spacesuit', 'A lonely character, sitting on a rock, staring at the sky, surrounded by a foreign landscape', 'A character sitting on a makeshift bed, staring at a family photo, while surrounded by space gear and futuristic technology', 'A character reminiscing, lying on a bed, with a picture of their dog, surrounded by equipment of a space shuttle', 'A character lying on the ground, looking sad, with an empty dog collar and bowl beside them on a foreign planet', 'A man with a pensive expression holding a pen and paper, a spacecraft in the distance', 'A dog with a determined look standing in front of a difficult obstacle course', 'A spaceship launching into the galaxy, a letter attached to the side', 'A dog sniffing around a space station, a message in his mouth', 'A man laying down on the ground, surrounded by darkness, a dim light in the distance', 'Stranded man, desert, no water in sight', 'Man standing amidst rocks, distant shining spacecraft, his only hope', 'A dog inside a spacecraft, looking out of the window, showing loyalty and happiness', 'Dog sniffing the ground, tracking, searching', 'A reunion between a man and his dog, spacecraft hatch open, hugging in joy', 'A person and a dog, in a prison cell, with a tunneling tool', 'A spacecraft, with two people inside, blasting off into space', 'Homecoming celebration, a crowd of people, balloons, and streamers', 'A person sitting on a rock, looking out at a scenic scene, with a dog sitting next to him', 'A person standing in front of a gravestone, with a dog by their side, looking somber', 'A couple holding hands, standing on a beach, with a beautiful sunset in the background']
    storyteller.story2audio()
    storyteller.scene2image()
    storyteller.image2video()
    
    # TODO: remove audio and images after video is made


if __name__=="__main__": main()