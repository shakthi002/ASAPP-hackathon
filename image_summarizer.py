# image_summarizer.py
import base64
from groq import Groq

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to summarize the images
def summarizer(image_paths):
    text = []  # Initialize text list to store summaries
    groq_api_key = "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"

    for image_path in image_paths:
        base64_image = encode_image(image_path)

        client = Groq(api_key=groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize the plot and graph diagram present in this image. Give me only a short and crisp summary. give me in less than 150 words."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
        )
        text.append(chat_completion.choices[0].message.content)
    #print(text)
    return text  # Return the list of summaries
