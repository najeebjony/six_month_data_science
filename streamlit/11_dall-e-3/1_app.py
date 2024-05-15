

import streamlit as st
import openai
import requests
from io import BytesIO

# Set your OpenAI API key
openai.api_key = "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"

# Title of the app
st.title("DALL-E 3 Image Generator Create BY Najeeb")

# Function to generate images
def generate_image(prompt, num_images=1):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
       #  create a  4 images at one time
        n=num_images,  # number of pictures
        size="1024x1024"
    )
    return response

# Function to download an image from a URL
def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:  # HTTP status code 200 means OK/success
        return BytesIO(response.content)

# Main function where the app functionality is implemented
def main():
    st.title("Generate Multiple Images with DALL-E 3")
    
    prompt = st.text_area("Enter your prompt")
    num_images = st.number_input("How many images to generate (1-4)?", min_value=1, max_value=4, value=1, step=1)
    
    if st.button("Generate Image(s)"):
        with st.spinner("Generating..."):
            response = generate_image(prompt, num_images)
            
            if response and 'data' in response and len(response['data']) > 0:
                for idx, img in enumerate(response['data']):
                    image_url = img['url']
                    st.image(image_url, caption=f'Generated Image {idx + 1}', use_column_width=True)

                    image_buffer = download_image(image_url)
                    if image_buffer:
                        st.download_button(
                            label="Download Image",
                            data=image_buffer,
                            file_name=f"generated_image_{idx + 1}.png",
                            mime="image/png"
                        )
            else:
                st.error('No images were generated.')

if __name__ == "__main__":
    main()
