from typing import List


def sentence_prompt(sentences:  List[str]):
    prompt = """
It is your job to generate a prompt for an image generating model.
The prompt that you generate should be descriptive and vivid and you can be creative if you want to.
Don't use any specific details of the story, such as names of characters or places, because the image generating model won't know what they are.
Only describe the elements in the physical environment that should be present in the image.
Try to use much keywords as possible. You are discouraged from using full sentences, but are allowed to in case only keywords are not descriptive enough.
Here are some examples, but don't generalise the topics from the examples. Only complete the prompt after END EXAMPLE.

START EXAMPLE

[Text 0] For months, Sarah had trained for this mission, preparing for every possible scenario and mastering the skills she would need to survive in the harsh environment of space.
[Text 1] John was a farmer who had lived on the same land his entire life.
[Text 2] The book became a bestseller and inspired generations to come to pursue their dreams, no matter how daunting the challenge may seem.
[Text 3] One day, while Max was on duty at the fire station, the alarm rang.
[Text 4] They had saved everyone who was trapped inside, and the neighborhood cheered as they emerged from the burning house.

[Prompt 0] A girl, training, space ship simulator
[Prompt 1] Strong male, farmer, in front of barn, tractor, farm dog
[Prompt 2] A person holding a book, standing in front of a large audience, with a backdrop of bright lights and applause
[Prompt 3] A firefighter, an alarm going off, a fire station
[Prompt 4] A happy community, a burning house, a firefighter

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
You don't even have to use all of the topics yet, but you can if you want to!

Write the first paragraph for a story about the following topics: {topics_str}.
        """
    else:
        prompt = f"""
You are an extremely creative writer that makes short stories for children.
You are renowned for your ability to make stories that are both entertaining and educational.
You add a lot of details about characters and events in the story.

Write the next paragraph for a story about the following topics: {topics_str}.

The previous paragraphs were:
{story}
        """

    return prompt