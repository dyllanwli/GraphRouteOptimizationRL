#!/bin/bash 
pip install --user tf-agents[reverb] 
# fore reverb https://github.com/deepmind/reverb 

sudo apt-get install -y xvfb ffmpeg freeglut3-dev
pip install 'imageio==2.4.0'
pip install pyvirtualdisplay
pip install tf-agents[reverb]
pip install pyglet
pip install pygame==2.1.2

pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html