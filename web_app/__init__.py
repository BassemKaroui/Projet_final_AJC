from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os
import pandas as pd
import torch


def create_app():

    #-----------------------------------------------------------------------------------#
    # INITIALISATION DE L'APPLICATION                                                   #
    #-----------------------------------------------------------------------------------#
    app = Flask(__name__)

    #-----------------------------------------------------------------------------------#
    # PAGES                                                                             #
    #-----------------------------------------------------------------------------------#

    # @requiredLogin

    @app.route('/')
    def homePage():
        return render_template("index.html")

    #-----------------------------------------------------------------------------------#
    # APIs                                                                              #
    #-----------------------------------------------------------------------------------#

    return app
