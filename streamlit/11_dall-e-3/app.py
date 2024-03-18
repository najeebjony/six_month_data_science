import streamlit as st
import openai
import requests
from io import BytesIO


# set your openai api key
openai.api_key = "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"

# title of app 
st.title("@Najeebullah")
# function to generate image
def generate_image(prompt):
    response = openai.Image.create(
        model = "dall-e-3",
        prompt=prompt,
        n=1, # number of pictures
        size="1024x1024"
    )
    return response
def downlaod_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 250:
        return BytesIO(response.content)
    
def main():
    st.title("DALL-E 3 Image Generator")
    # st.write("This is a demo of OpenAI's DALL-E 3 model. It can generate images based on the prompt you provide.")
    
    prompt = st.text_area("Enter your prompt")
    if st.button("Generate Image"):
        with st.spinner("Generating..."):
            response = generate_image(prompt)
            if response and 'date' in response and len(response.data) > 0:
                image_url = response.data[0].url
                st.image(image_url, caption='Generated Image', use_column_width=True)


                #downlaod functionality 
                image_buffer = downlaod_image(image_url)
                if image_buffer:
                    st.download_button(
                        label="Download Image",
                        data=image_buffer,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
            else:
                st.error('No images generated')
if __name__ == "__main__":
    main()
                
    