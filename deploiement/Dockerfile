FROM continuumio/miniconda3

RUN conda create -n app_env pytorch torchvision torchaudio cudatoolkit=11.1 flask pandas numpy matplotlib -c pytorch -c nvidia -c anaconda -c conda-forge
RUN echo "source activate app_env" > ~/.bashrc
ENV PATH /opt/conda/envs/app_env/bin:$PATH

ENV FLASK_APP web_app
ENV FLASK_ENV=development

EXPOSE 5000

WORKDIR /root

COPY . .
