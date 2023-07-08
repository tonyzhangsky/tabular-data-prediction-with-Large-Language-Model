import warnings
warnings.filterwarnings("ignore")

#Math and Vectors
import pandas as pd
import numpy as np

#Visualizations
import plotly.graph_objects as go

#ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.classifier import ROCAUC
from sklearn import metrics
import openai
import json
import os

# OpenAI API key setting
openai.api_key = os.getenv('OPENAI_API_KEY')

## define function
def ml_models():
    lr = LogisticRegression(penalty='none', solver='saga', random_state=42, n_jobs=-1)
    lasso = LogisticRegression(penalty='l1', solver='saga', random_state=42, n_jobs=-1)
    ridge = LogisticRegression(penalty='l2', solver='saga', random_state=42, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50, 
                                max_features=0.3, random_state=42, n_jobs=-1)
    models = {'LR': lr, 'LASSO': lasso, 'RIDGE': ridge, 'RF': rf}
    return models


def prediction_GPT3_5(data, explain = False):
    if explain:
        prompt = '''You are a medical expert / underwriter in a global insurance company. Your job is to evaluate the chance of having heart attack.
        Please encode your response as json in the following format
        {{
            "decision": "<Either less chance of heart attack or more chance of heart attack>",
            "reasoning": "<a 10-30 plain word description of why you made this decision>"
        }}
        ---- BEGIN OF THE DATA ----
        What is the age of the applicant?: {age}
        What is the sex of the applicant?: {sex}
        What is the chest pain type of the applicant?: {cp}
        What is the resting blood pressure (in mm Hg) of the applicant?: {trtbps}
        What is the cholestoral in mg/dl fetched via BMI sensor of the applicant?: {chol}
        What is the fasting blood sugar level of the applicant?: {fbs}
        What is the resting electrocardiographic results of the applicant?: {restecg}
        What is the maximum heart rate achieved of the applicant?: {thalachh}
        What is the exercise induced angina of the applicant?: {exng}
        What is the ST depression induced by exercise relative to rest of the applicant?: {oldpeak}
        What is the slope of the peak exercise ST segment of the applicant?: {slp}
        What is the number of major vessels of the applicant?: {caa}
        What is the thall of the applicant?: {thall}
        ---- END OF THE DATA ----\n'''.format(**data)
    else:    
        prompt = '''You are a medical expert / underwriter in a global insurance company. Your job is to evaluate the chance of having heart attack.
        Please encode your response as json in the following format
        {{
            "decision": "<Either less chance of heart attack or more chance of heart attack>",
        }}
        ---- BEGIN OF THE DATA ----
        What is the age of the applicant?: {age}
        What is the sex of the applicant?: {sex}
        What is the chest pain type of the applicant?: {cp}
        What is the resting blood pressure (in mm Hg) of the applicant?: {trtbps}
        What is the cholestoral in mg/dl fetched via BMI sensor of the applicant?: {chol}
        What is the fasting blood sugar level of the applicant?: {fbs}
        What is the resting electrocardiographic results of the applicant?: {restecg}
        What is the maximum heart rate achieved of the applicant?: {thalachh}
        What is the exercise induced angina of the applicant?: {exng}
        What is the ST depression induced by exercise relative to rest of the applicant?: {oldpeak}
        What is the slope of the peak exercise ST segment of the applicant?: {slp}
        What is the number of major vessels of the applicant?: {caa}
        What is the thall of the applicant?: {thall}
        ---- END OF THE DATA ----\n'''.format(**data)
    print(prompt)
    response = openai.Completion.create(
        #model ='gpt-3.5-turbo',
        model = 'text-davinci-003',
        prompt=prompt,
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    try:
        output = response.choices[0].text.strip()
        output_dict = json.loads(output)
        return output_dict
    except (IndexError, ValueError):
        return None

def prediction(combined_data_argu):
    credit_data, explain = combined_data_argu
    response = prediction_GPT3_5(credit_data, explain)
    return response

def create_auc_chart(scores_dic, label_y_test, title):

    fig = go.Figure()
    trace_list = []
    
    for key in scores_dic:
        
        y_pred = scores_dic[key]
        fpr, tpr, _ = metrics.roc_curve(label_y_test, y_pred)
        auc = round(metrics.roc_auc_score(label_y_test, y_pred), 4)
        
 
        trace_tmp = go.Scatter(
            x=fpr,
            y=tpr,
            #text=df.index,
            name=key + ", AUC = " + str(auc),
            #marker_color='#f5827a',
            hovertemplate='FPR: %{x:.2%} TPR: %{y:.2%}',
            showlegend=True

        )
        trace_list.append(trace_tmp)
 

    fig.add_traces(trace_list)

    fig.update_layout(dict(
        title=dict(text=title.upper()),
        template='plotly_white',
        title_font_family="Times New Roman",
        font_family="Courier New",   
        height=600,   
        width=960,
    ))
    
    return fig

def compile_prompt(x):
    pattern = r'\[(\d+)-(\d+)\)'
    prompt = (
        f"The age of the applicant is {x['age']}. "
        f"The sex of the applicant is {x['sex']}. "
        f"The chest pain type of the applicant is {x['cp']}. "
        f"The resting blood pressure (in mm Hg) of the applicant is {x['trtbps']}. "
        f"The cholestoral in mg/dl fetched via BMI sensor of the applicant is {x['chol']}. "
        f"The fasting blood sugar level of the applicant is {x['fbs']}. "
        f"The resting electrocardiographic results of the applicant is {x['restecg']}. "
        f"The maximum heart rate achieved of the applicant is {x['thalachh']}. "
        f"The exercise induced angina of the applicant is {x['exng']}. "
        f"The ST depression induced by exercise relative to rest of the applicant is {x['oldpeak']}. "
        f"The slope of the peak exercise ST segment of the applicant is {x['slp']}. "
        f"The number of major vessels of the applicant is {x['caa']}. "
        f"The thall of the applicant is {x['thall']}. "
    )
    return prompt

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


## ChatGPT prediction
# Define the message objects
def gpt_reasoning(data):
    prompt = '''You are a medical expert / underwriter in a global insurance company. Your job is to evaluate the chance of having heart attack.
        Please encode your response as json in the following format
        {{
            "decision": "<Either less chance of heart attack or more chance of heart attack>",
            "reasoning": "<Provide a 300 words explaination of why you made this decision>"
        }}
        ---- BEGIN OF THE DATA ----
        What is the age of the applicant?: {age}
        What is the sex of the applicant?: {sex}
        What is the chest pain type of the applicant?: {cp}
        What is the resting blood pressure (in mm Hg) of the applicant?: {trtbps}
        What is the cholestoral in mg/dl fetched via BMI sensor of the applicant?: {chol}
        What is the fasting blood sugar level of the applicant?: {fbs}
        What is the resting electrocardiographic results of the applicant?: {restecg}
        What is the maximum heart rate achieved of the applicant?: {thalachh}
        What is the exercise induced angina of the applicant?: {exng}
        What is the ST depression induced by exercise relative to rest of the applicant?: {oldpeak}
        What is the slope of the peak exercise ST segment of the applicant?: {slp}
        What is the number of major vessels of the applicant?: {caa}
        What is the thall of the applicant?: {thall}
        ---- END OF THE DATA ----\n'''.format(**data)
    print(prompt)
    message_objects = [
        {"role": "system", "content": '''You are a medical expert / underwriter in a global insurance company. Your job is to evaluate the chance of having heart attack. Please encode your response as json in the following format
        {{
            "decision": "<Either less chance of heart attack or more chance of heart attack>",
        }}'''},
        {"role": "user", "content": prompt},
    ]

    # Make the API call
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_objects,
        max_tokens=1000,  # Adjust the max_tokens as per your desired response length
        stop=None,  # Set custom stop conditions if required
    )

    # Extract the response message content
    response_content = completion.choices[0].message["content"]
    # print("Response:", response_content)

    return response_content