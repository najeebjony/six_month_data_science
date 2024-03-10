import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import PyPDF2
import seaborn as sns
import numpy as np
from docx import Document
import plotly.express as px
import base64 
from io import BytesIO
import base64
from io import BytesIO

# function for the reading 
def read_txt(file):
    return file.getvalues().decode('utf-8')

# function for the reading
def read_docx(file):
    doc = Document(file)
    return"".join([p.text for p in doc.paragraphs])

def read_pdf(file):
    pdf = PyPDF2.PdfFileReader(file)
    return "".join([page.extractText() for page in pdf.pages])

# function to filter out stopword 
def remove_stopwords(text, additional_stopwords=[]):
    words = text.split()
    all_stopword = STOPWORDS.union(set(additional_stopwords))
    filtered_words = [word for word in words if word.lower() not in all_stopword]
    return "".join(filtered_words)

# function to create download link for plot 
def gey_download_link(buffered, format_):
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/{format_};base64,{image_base64}" download="plot.{format_}">Download {format_}</a>'

# function to generate a download link for a dataframe
def generate_download_link(df, filename,file_label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return  f'<a href="data:file/csv;base64,{b64}" download="{filename}">{file_label}</a>'

# streamlit code 
st.title("Word Cloud Generator")
st.subheader("Upload a pdf, docx or txt file to generate a word cloud")

uploaded_file = st.file_uploader("Choose a file", type=["pdf","docx","txt"])
st.set_option("deprecation.showPyplotGlobalUse", False)

if uploaded_file:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)
    
if uploaded_file.type == "text/plain":
    text = read_txt(uploaded_file)

elif uploaded_file.type == "application/pdf":
    text = read_pdf(uploaded_file)

elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    text = read_docx(uploaded_file)
else:
    st.error("file Type not supported. Please upload a pdf, docx or txt file")
    st.stop()


# Generate word count table 
words = text.split()
word_count = pd.DataFrame ({'word':words}).groupby('word').size().reset_index(name='count').sort_index(ascending=False)

#slidebar: chechbox and multiselect box for stopwords 
use_stander_stopwords = st.sidebar.checkbox("Use standard stopwords",True)
top_words = word_count['word'].head(50).tolist()
additional_stopwords = st.sidebar.multiselect("Additional stopwords",sorted(top_words))

if use_stander_stopwords:
    all_stopwords = STOPWORDS.union(set(additional_stopwords))
else:
    all_stopwords = set(additional_stopwords)

text = remove_stopwords (text, all_stopwords)

if text:

    #word cloud dimensions 
    width = st.sidebar.slider("Select Word Cloud Width", 400, 2000, 1200,50)
    height = st.sidebar.slider("Select Word Cloud Height", 400, 2000, 800,50)

    # Generate word cloud

    st.subheader("Word Cloud")
    fig, ax = plt.subplots(figsize= (width/100, height/100))
    wordcloud = WordCloud(width=width, height=height, background_color="white", max_words=100).generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    # save plot functionality 
    format =  st.selectbox("Select file format", ["png", "jpg", "svg", "pdf"])
    resolution = st.slider("Select resolution", 100, 500, 300,50)

    # generate word count table
    st.subheader("Word Count Table")
    word = text.split()
    word_count = pd.DataFrame ({'word':word}).groupby('word').size().reset_index(name='count').sort_index(ascending=False)
    st.write(word_count)
# function to generate download link for image
def get_image_download_link(buffered, format_):
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/{format_};base64,{image_base64}" download="plot.{format_}">Download {format_}</a>'

st.pyplot(fig)
if st.button(f"Save as {format}"):
    buffered = BytesIO()
    plt.savefig(buffered, format=format, dpi=resolution)
    st.markdown(get_image_download_link(buffered, format), unsafe_allow_html=True)

# download word count table
st.sidebar.markdown("==")
st.sidebar.subheader("Download")

# add a facebook , github , linkedin and twitter link to the sidebar
st.sidebar.markdown("### Social Media Links")
st.sidebar.markdown("[Facebook](https://www.facebook.com/)")
st.sidebar.markdown("[GitHub](https://github.com/)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/)")
st.sidebar.markdown("[kaggle](https://www.kaggle.com/)")

# download word count table
st.subheader("Word Count Table")
st.write(word_count)
st.markdown(get_image_download_link(word_count, "word_count.csv", "Click Here to Download"), unsafe_allow_html=True)



