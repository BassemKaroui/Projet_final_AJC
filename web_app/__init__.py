from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import send_file
from flask import redirect
from flask import url_for
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

val_df = pd.read_csv('web_app/Human_protein_atlas/val.csv', nrows=25)


def create_app():

    #-----------------------------------------------------------------------------------#
    # INITIALISATION DE L'APPLICATION                                                   #
    #-----------------------------------------------------------------------------------#

    UPLOAD_FOLDER = 'web_app/static/uploads/'
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    #-----------------------------------------------------------------------------------#
    # Modèle                                                                            #
    #-----------------------------------------------------------------------------------#

    device = torch.device(torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'))
    use_amp = True

    class ModelWithAttention(nn.Module):

        def __init__(self):
            super().__init__()
            resnext = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
            self.resnext = nn.Sequential(*list(resnext.children())[:-2])
            self.resnext.requires_grad_(False)
            self.attention_nn = nn.Sequential(
                nn.Linear(2048, 10),
                nn.Softmax(dim=1)
            )
            self.multi_label_classifier = nn.Conv1d(
                10, 10, kernel_size=2048, groups=10)
            self.top = nn.ModuleList(
                [self.attention_nn, self.multi_label_classifier])

        @amp.autocast(enabled=use_amp)
        def forward(self, imgs):
            # shape : batch_size x 2048 x H x W
            encoded_imgs = self.resnext(imgs)
            # shape: batch_size x (HxW) x 2048
            encoded_imgs = encoded_imgs.reshape(
                *encoded_imgs.shape[:2], -1).swapaxes(1, 2)
            # shape: batch_size x (HxW) x 10
            weights = self.attention_nn(encoded_imgs)
            encoded_imgs = encoded_imgs.unsqueeze(dim=1).repeat(
                1, 10, 1, 1)  # shape: batch_size x 10 x (HxW) x 2048
            weights = weights.swapaxes(1, 2).unsqueeze(
                dim=-1)  # shape: batch_size x 10 x (HxW) x 1
            # shape: batch_size x 10 x (HxW) x 2048
            outputs = weights * encoded_imgs
            outputs = outputs.sum(dim=2)  # shape: batch_size x 10 x 2048
            # shape: batch_size x 10 x 1 => batch_size x 10 (after squeezing)
            outputs = self.multi_label_classifier(outputs).squeeze()
            return outputs, weights

    model = ModelWithAttention()
    model.to(device)
    model.load_state_dict(torch.load(
        'web_app/model_checkpoints/model_epoch_32.pth'))

    thresholds = torch.tensor([0.866, 0.28, 0.95, 0.27599999, 0.52200001,
                              0.45899999, 0.68699998, 0.81699997, 0.75999999, 0.61299998], device=device)

    def visualize_att_mask(img_path, model, root_path, device=device, threshold=thresholds):

        tmp_files = os.listdir(root_path)
        for file in tmp_files:
            path = os.path.join(root_path, file)
            if os.path.isfile(path):
                os.remove(path)

        img = Image.open(img_path).convert('RGB')
        img_to_tensor = transforms.ToTensor()
        img = img_to_tensor(img)
        img = img.unsqueeze(dim=0).to(device)  # shape : 1 x 3 x 512 x 512
        with torch.no_grad():
            with amp.autocast(enabled=use_amp):
                model.eval()
                logits, weights = model(img)
                probs = torch.sigmoid(logits)
        labels = probs >= threshold
        labels = torch.arange(10)[labels]
        if labels.shape == (0,):
            labels = probs.argmax(dim=-1, keepdim=True)
        labels = labels.cpu()
        weights = weights.squeeze()[labels].unsqueeze(
            dim=0).reshape(1, labels.shape[0], 16, 16).cpu()
        upsampled_weights = F.upsample(weights, size=512, mode='bilinear')
        img = img.cpu()
        for i, protein_idx in enumerate(labels):
            idx = protein_idx.item()
            fig = plt.figure(figsize=(13, 13))
            plt.imshow(img[0].permute(1, 2, 0), cmap='Greys_r')
            plt.imshow(upsampled_weights[0, i, :, :],
                       cmap='Greys_r', alpha=0.6)
            plt.axis('off')
            plt.savefig(os.path.join(
                root_path, f'protein_{idx}.png'), bbox_inches='tight')
            plt.close(fig)
        return probs.tolist(), labels.tolist()
    #-----------------------------------------------------------------------------------#
    # PAGES                                                                             #
    #-----------------------------------------------------------------------------------#

    @app.route('/')
    def homePage():
        return render_template("index.html")

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            if 'client_img' not in request.files:
                return 'there is no client_img in form!'
            clientImg = request.files['client_img']
            path = os.path.join(
                app.config['UPLOAD_FOLDER'], clientImg.filename)
            clientImg.save(path)
            messages = json.dumps({"main": clientImg.filename})
            return redirect(url_for("homePage", messages=messages))

            return 'ok'
        return '''
        <h1>Upload new File</h1>
        <form method="post" enctype="multipart/form-data">
        <input type="file" name="client_img">
        <input type="submit">
        </form>
        '''

    @app.route('/images/<image>')
    def get_image(image):
        if image[0] == 'p':
            filename = f'tmp_predictions/{image}'
        else:
            filename = f'Human_protein_atlas/train/{image}'
        return send_file(filename, mimetype='/images/png')

    @app.route('/uploads/<image>')
    def get_uploads(image):
        filename = f'static/uploads/{image}'
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
        data = request.json  # {'Image': ____}
        img_path = os.path.join(
            'web_app/Human_protein_atlas/train', str(data['Image'])+'.png')
        _, labels = visualize_att_mask(
            img_path, model, 'web_app/tmp_predictions', threshold=thresholds)

        return {"classes": labels}

    return app
