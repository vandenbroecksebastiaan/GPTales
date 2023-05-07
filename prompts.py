from typing import List


def sentence_prompt(sentences:  List[str]):
    prompt = """
It is your job to generate a prompt for an image generating model.
The prompt that you generate should be descriptive and vivid and you can be creative if you want to.
Don't use any specific details of the story, such as names of characters or places, because the image generating model won't know what they are.
Only describe the elements in the physical environment that should be present in the image.
Try to use much keywords as possible. You are discouraged from using full sentences, but are allowed to in case only keywords are not descriptive enough.
Include all information between brackets, but don't include the brackets themselves.
Here are some examples, but don't generalise the topics from the examples. Only complete the prompt after END EXAMPLE.

START EXAMPLE

[Text 0] For months, Sarah had trained for this mission, preparing for every possible scenario and mastering the skills she would need to survive in the harsh environment of space.
[Text 1] John was a farmer who had lived on the same land his entire life.
[Text 2] as the days passed, [a friendly dog with brown and white fur] became more than just a farm dog to [humans with advanced technology and resourceful skills]. 
[Text 3] [a curious girl with shoulder-length brown hair, wearing a green dress and brown boots] was not afraid of the [a small, gray mouse with big brown eyes] and approached it with a welcoming smile. 
[Text 4] the [delicate creatures with wings that sparkled in the sunlight and wore dresses made of the softest petals] urged [a young girl with a curious spirit and a brave heart] to approach it, assuring her that it was the key to restoring balance.


[Prompt 0] A girl, training, space ship simulator
[Prompt 1] Strong male, farmer, in front of barn, tractor, farm dog
[Prompt 2] A friendly dog with brown and white fur, humans with advanced technology
[Prompt 3] A girl with shoulder-length brown hair wearing a green dress and brown boots, a small gray mouse with big brown eyes and a smile
[Prompt 4] A sparkling creature wearing a dress, a young girl

END EXAMPLE

    """
    for idx, sentence in enumerate(sentences): prompt = prompt + f"\n[Text {idx}] {sentence}"
    return prompt

def story_prompt(topics: List[str], story: str = ""):
    if len(topics) == 1: topics.append("")
    topics_str = topics[0] + "".join([", " + i for i in topics[1:]])
    if story == "":
        prompt = f"""
You are an extremely creative writer that makes short stories for children.
You are renowned for your ability to make stories that are both entertaining and educational.
You add a lot of details about characters and events in the story.
Remember, this is only the first paragraph of the story, so don't give away too much information.
Capitalize words that should have an emphasis.
You don't even have to use all of the topics yet, but you can if you want to!
Do not end the story. Only write the introduction.

Write the first paragraph for a story about the following topics: {topics_str}.
        """
    else:
        prompt = f"""
You are an extremely creative writer that makes short stories for children.
You are renowned for your ability to make stories that are both entertaining and educational.
You add a lot of details about characters and events in the story.
Capitalize words that should have an emphasis.
You don't even have to use all of the topics yet, but you can if you want to!

Write the next paragraph for a story about the following topics: {topics_str}.

The previous paragraphs were:
{story}
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