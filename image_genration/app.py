from flask import Flask , render_template, request, redirect, url_for
import openai 

# create a flask app 
app = Flask(__name__) 

# load your api key from an environment variable
openai.api_key = "YOUR_API_KEY"

def generate_image(prompt):
    clinet = openai.OpenAI(api_key=openai.api_key)
    response = clinet.images.generate(
        model = "dall-e-3"
        prompt = prompt,
        size = "1024x1024",
        quality = "standard"
        n = 1 
    )
    return response [0].url
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']

        if prompt:
            image_url = generate_image(prompt)
            if image_url:
                return render_template('index.html', image_url = image_url)
            else:
                return render_template('index.html', error = "Error generating image please try again")
        else:
            return render_template('index.html', error = "Please enter a description")
        return render_template('index.html')

if __name__== '__main__':
    app.run(debug=True)