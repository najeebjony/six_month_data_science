from flask import Flask, render_template_string
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# HTML Template
base_template = """
<!doctype html>
<html>
<head>
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>{{ title }}</h1>
    <div>{{ content|safe }}</div>
    <hr>
    <a href="/">Home</a>
    <a href="/survival_analysis">Survival Analysis</a>
    <a href="/class_distribution">Class Distribution</a>
    <a href="/age_distribution">Age Distribution</a>
</body>
</html>
"""

@app.route('/')
def home():
    content = "<p>Welcome to the Titanic Data Analysis App. Choose an analysis:</p>"
    return render_template_string(base_template, title='Titanic Dataset Analysis', content=content)

@app.route('/survival_analysis')
def survival_analysis():
    fig = px.pie(titanic, names='survived', title='Survival Rate')
    graph_html = pio.to_html(fig, full_html=False)
    return render_template_string(base_template, title='Survival Analysis', content=graph_html)

@app.route('/class_distribution')
def class_distribution():
    fig = px.histogram(titanic, x='class', color='survived', barmode='group', title='Passenger Class Distribution')
    graph_html = pio.to_html(fig, full_html=False)
    return render_template_string(base_template, title='Class Distribution', content=graph_html)

@app.route('/age_distribution')
def age_distribution():
    fig = px.histogram(titanic, x='age', color='survived', title='Age Distribution', nbins=20)
    graph_html = pio.to_html(fig, full_html=False)
    return render_template_string(base_template, title='Age Distribution', content=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
