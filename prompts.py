from typing import List


def sentence_prompt(sentences: List[str]):
    prompt = """
It is your job to generate a prompt for an image generating model.
The prompt that you generate should be descriptive and vivid and you can be creative if you want to.
Only describe the elements in the physical environment that should be present in the image.
Try to use much keywords as possible. You are discouraged from using full sentences, but are allowed to in case only keywords are not descriptive enough.
Include all information between brackets, but don't include the brackets themselves.
Here are some examples, but don't generalise the topics from the examples. Only complete the prompt after END EXAMPLE.

START EXAMPLE

[Text 0] For months, Sarah had trained for this mission, preparing for every possible scenario and mastering the skills she would need to survive in the harsh environment of space.
[Text 1] this was just the beginning, but with their new hope, she knew that a brighter future was possible. 
[Text 2] as the days passed, [a friendly dog with brown and white fur] became more than just a farm dog to [humans with advanced technology and resourceful skills]. 
[Text 3] [a curious girl with shoulder-length brown hair, wearing a green dress and brown boots] was not afraid of the [a small, gray mouse with big brown eyes] and approached it with a welcoming smile. 
[Text 4] the [delicate creatures with wings that sparkled in the sunlight and wore dresses made of the softest petals] urged [a young girl with a curious spirit and a brave heart] to approach it, assuring her that it was the key to restoring balance.


[Prompt 0] A white girl with brown hair, practicing and training, space ship simulator
[Prompt 1] Twinkling lights, a new hope
[Prompt 2] A friendly dog with brown and white fur, humans with advanced technology
[Prompt 3] A girl with shoulder-length brown hair wearing a green dress and brown boots, a small gray mouse with big brown eyes and a smile
[Prompt 4] A sparkling creature wearing a dress, a young girl

END EXAMPLE

    """
    for idx, sentence in enumerate(sentences): prompt = prompt + f"\n[Text {idx}] {sentence}"
    return prompt

def story_prompt(topics: List[str], option: int = None, story: str = ""):
    if len(topics) == 1: topics.append("")
    topics_str = topics[0] + "".join([", " + i for i in topics[1:]])
    if story == "":
        prompt = f"""
You are an extremely creative writer that makes short stories for children.
You are renowned for your ability to make stories that are both entertaining and educational.
You add a lot of details about characters and events in the story.
It is your task to write the first (long) paragraph of a story about the following topics: {topics_str}.
Capitalize words that should have an emphasis.
You don't even have to use all of the topics yet, but you can if you want to!
After the paragraph, provide two options for the next direction in the story for the reader to choose from.
Do not end the story. Only write the introduction.

BEGIN EXAMPLE
In the heart of the forest, there lived a BEAR named Benny. Unlike other bears who spent their days searching for honey and catching fish, Benny had a unique HOBBY. He loved to collect trinkets that he found in the forest, from shiny rocks to colorful feathers. One day, Benny had a bright idea. He would open a STORE to showcase his collection to the other animals in the forest. And so, with determination and hard work, BENNY'S BEAR STORE was born. On the GRAND OPENING DAY, all the creatures in the forest were invited to see what Benny had to offer.

[1] A famous animal came to the store and bought a trinket.
[2] A bad animal came to the store and Benny had to defend the store.
END EXAMPLE
        """
    else:
        prompt = f"""
You are an extremely creative writer that makes short stories for children.
You are renowned for your ability to make stories that are both entertaining and educational.
You add a lot of details about characters and events in the story.
It is your task to continue writing the story about the following topics: {topics_str}.
Capitalize words that should have an emphasis.
You don't even have to use all of the topics yet, but you can if you want to!
After the paragraph, provide two options for the next direction in the story for the reader to choose from.
Do not end the story. Only write the introduction.

BEGIN EXAMPLE
In the vast expanse of space, an ASTRONAUT named Alex floated alone. At first, it was exhilarating to be so far away from everything she had ever known, nothing but the stars and galaxies to keep her company. But as the hours turned into days, and the days into weeks, the excitement turned into fear. Alex's spacecraft malfunctioned, sending her hurtling further into the void. She tried every trick she knew, but nothing seemed to work. Her only comfort in that lonely, endless expanse was her trusty DOG companion, Ranger. Ranger was a smart and loyal dog, trained by Alex to be her co-pilot on her many journeys. As the days passed, Alex began to worry that she would never get home, but Ranger never left her side, even as the oxygen ran lower and lower. 

[1] Alex and Ranger come across a mysterious planet.
[2] Alex must fix the ship, but Ranger gets separated from her.

Continue the story with option 1.

Alex and Ranger floated through the inky blackness, the only light coming from the distant stars and the occasional flash of a meteor. After weeks of drifting through space, they finally spotted a planet on the horizon. It was small and rocky, with no discernible signs of life, but Alex decided it was worth investigating. She navigated her way towards the planet's surface, with Ranger howling with excitement at the new adventure. As they landed, they noticed something odd about the terrain- it seemed to glow a faint shade of green. Alex reached down to take a sample, but before she could, they were surrounded by a group of strange, insect-like creatures. The creatures chittered and buzzed in excitement, and Alex realized that they were intelligent beings. With Ranger by her side, they began to communicate with the creatures, learning about their society, their customs, and their way of life.

[1] Alex and Ranger learn about the creatures' culture.
[2] Alex and Ranger must find a way to escape the planet.

END EXAMPLE

{story}

Continue the story with option {option}.
        """
    return prompt

def story_clean_prompt(paragraph: str):
    prompt = f"""
You are a highly intelligent bot that is able to recognize characters from paragraph of text.
You are very creative and are able to add new details about the physical appearance of the characters and places.
Add the skin color, height, hair color, clothing, and any other details that you can think of.
If you don't know any specific details, you can make them up.
Do not forget to change the pronouns if necessary.
Find all characters from the following text and write them down in the format: "character": "description".

START EXAMPLE

"Max": "a tall, white boy with a mustache and brown hair, wearing jeans and a t-shirt"
"Lena": "a short, black girl with curly hair and glasses, wearing a blue dress and a necklace"
"house": "a large, two-story house with a red roof"
"farm": "a small farm with a barn, a farmhouse and a tractor"

END EXAMPLE

{paragraph}
    """
    return prompt
    
"""
START EXAMPLE

[PROMPT]
The dog, now named Ehro, quickly adapted to life at the police station.
He spent his days playing with the officers, sniffing out clues, and even helping with the occasional search and rescue mission.
The kids in the neighborhood loved to come visit Ehro and take him for walks around the block.
But his biggest moment of heroism came one fateful day when a dangerous criminal escaped from his holding cell.
Without hesitation, Ehro sprang into action, barking loudly and alerting the officers to the criminal's location.
Thanks to Ehro's bravery, the criminal was quickly recaptured and brought back to his cell.
From that day on, Ehro was known as not just a lovable pup, but also a true hero of the police station. 
[RESPONSE]
Ehro: heroic dog

END EXAMPLE

START EXAMPLE

[PROMPT]
On a beautiful sunny day, down on the farm, a sleek black and white cat named Socks was lounging lazily on the warm hay in the barn.
She watched as the farmers bustled about, tending to the animals and equipment.
Every now and then, a curious chicken or two would wander in through the open door, and Socks would arch her back and give them a disgusted look before settling back into her nap.
The barn was her kingdom and she ruled over it like a queen. But little did she know, today was going to be an adventure she would never forget. 
As Socks closed her eyes and drifted off to sleep, she heard a commotion outside the barn door.
Suddenly, a group of mischievous mice scampered in and began to raid the farmer's grain bins.
Socks's ears perked up, and she sprang into action, leaping and pouncing on the tiny rodents with lightning-fast reflexes.
As the mice scurried away defeated, Socks realized that she had just saved the farmer's crops from an infestation.
Feeling proud of herself, she curled up on top of a bale of hay and admired her kingdom once more, knowing that she had protected it with her feline prowess. 
[RESPONSE]
Socks: cat

END EXAMPLE

[PROMPT]
"""