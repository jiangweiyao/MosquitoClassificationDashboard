import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import PIL.Image as Image
from io import BytesIO
import base64


import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'The Mosquito Classifier'
server = app.server


device = torch.device('cpu')
model = torch.load('model_conv.pth', map_location=device)
labels = np.array(open("class.txt").read().splitlines())

preprocess = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

app.layout = html.Div([
    html.Div([
        html.H2('The Mosquito Classifier'),
        html.Strong('This application will attempt to classify your picture into 5 different species of mosquito or non-vector. Drag your image file into the below box to classify.'),
    ]),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename, date):
    #convert uploaded image file in Pillow image file
    encoded_image = contents.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    image = Image.open(bytes_image).convert('RGB')


    img = preprocess(image)
    img = img.unsqueeze(0)
    
    pred = model(img)

    print(pred.detach().numpy())

    df = pd.DataFrame({'class':labels, 'probability':pred[0].detach().numpy()})
    
    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        generate_table(df.sort_values(['probability'], ascending=[False]))
    ])

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(host='0.0.0.0',debug=True, port=8050)
