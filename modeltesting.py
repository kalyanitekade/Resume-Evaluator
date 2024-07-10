import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk
import plotly.io as pio

nltk.download('punkt')


def evaluateresume(jobdescription,resumepath):
    pdf = PyPDF2.PdfReader(resumepath)
    resume = ""
    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        resume += pageObj.extract_text()

    input_CV = preprocess_text(resume)
    input_JD = preprocess_text(jobdescription)

    # Model evaluation
    model = Doc2Vec.load('cv_job_maching.model')
    v1 = model.infer_vector(input_CV.split())
    v2 = model.infer_vector(input_JD.split())
    similarity = 100 * (np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    print(round(similarity, 2))

    # Visualization
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=similarity,
        mode="gauge+number",
        title={'text': "Matching percentage (%)"},
        # delta = {'reference': 100},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "#FFB6C1"},
                {'range': [50, 70], 'color': "#FFFFE0"},
                {'range': [70, 100], 'color': "#90EE90"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}}))

    fig.update_layout(width=600, height=400)  # Adjust the width and height as desired
    # fig.show()


    image_path = 'static/images/matching_percentage.png'
    pio.write_image(fig, image_path)


    # Print notification
    if similarity < 50:
        print("Low chance, need to modify your CV!", "red")
    elif similarity >= 50 and similarity < 70:
        print("Good chance but you can improve further!", "yellow")
    else:
        print("Excellent! You can submit your CV.", "green")

    print("Similarity is " , similarity)
    return {'result': image_path, 'score': str(similarity)}


def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)

    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text


# Apply to CV and JD


# JD by input text:

'''jd = """
Proactively engage with clients in the branch and deliver an outstanding service experience through completing transactions and by identifying opportunities for advice, solutions, digital enablement and partner introductions
Focus on education and demonstration, leverage technology to deliver a memorable client experience, drive solutions and retain business
Contribute to team results by listening and spotting opportunities to offer additional advice, introduce clients to the capability of RBC partners, or personally fulfil client solutions
Proactively take ownership of resolving and preventing client banking problems
Cultivate and maintain relationships with partners to work as one RBC team
Manage risks by adhering to compliance routines, processes, and controls to protect client and shareholder interests while completing transactions
What do you need to succeed?

Must-have

Goal-oriented individual with a demonstrated passion for putting clients first.
Drive and self-motivation, as well as excellent communication skills and emotional intelligence
Digital literacy across a broad range of devices (i.e., smartphones, tablets, laptops, etc.)
Personal flexibility to work flex hours
Eagerness to learn and determination to succeed
Confidence and ability to learn financial concepts and willingness to obtain the Investment Funds in Canada or the Canadian Securities Course
Nice-to-have

Track record in building rapport and maintaining client relationships within the financial, service or retail industry
Mutual Funds accreditation
Is this job right for you? Check out our video and decide for yourself!

What’s in it for you?

We thrive on the challenge to be our best, progressive thinking to keep growing, and working together to deliver trusted advice to help our clients thrive and communities prosper. We care about each other, reaching our potential, making a difference to our communities, and achieving success that is mutual.

A comprehensive Total Rewards Program including bonuses and flexible benefits, competitive compensation, commissions, and stock where applicable
A world-class training program in financial services
Excellent career development and access to a variety of job opportunities across business and geographies
Leaders who support your development through coaching and managing opportunities
Work in a dynamic, collaborative, progressive, and high-performing team
We also strive to provide an accessible candidate experience for our prospective employees with different abilities. Please let us know if you need any accommodations during the recruitment process.

Join our Talent Community

Stay in-the-know about great career opportunities at RBC. Sign up and get customized info on our latest jobs, career tips and Recruitment events that matter to you.

Expand your limits and create a new future together at RBC. Find out how we use our passion and drive to enhance the well-being of our clients and communities at rbc.com/careers.
"""'''





'''evaluateresume(jd,'./CV/Akshay_Srimatrix.pdf')'''
