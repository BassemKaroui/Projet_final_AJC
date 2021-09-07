from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import send_file
import os
import pandas as pd
import torch

val_df = pd.read_csv('web_app/Human_protein_atlas/val.csv', nrows=25)


def create_app():

    #-----------------------------------------------------------------------------------#
    # INITIALISATION DE L'APPLICATION                                                   #
    #-----------------------------------------------------------------------------------#
    app = Flask(__name__)

    #-----------------------------------------------------------------------------------#
    # PAGES                                                                             #
    #-----------------------------------------------------------------------------------#

    @app.route('/')
    def homePage():
        return render_template("index.html")

    @app.route('/images/<image>')
    def get_image(image):
        filename = f'Human_protein_atlas/train/{image}'
        return send_file(filename, mimetype='/images/png')

    #-----------------------------------------------------------------------------------#
    # APIs                                                                              #
    #-----------------------------------------------------------------------------------#

    @app.route('/api/get_images')
    def get_images():
        data = val_df[['Image', 'Label']].head(25).to_dict('list')
        return jsonify(data)

    @app.route('/api/predict', methods=['POST'])
    def predict():
        # TO DO
        return {"classes": [4, 7]}

    return app
