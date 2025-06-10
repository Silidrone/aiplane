# Introduction

AiPlane is an AI agent (reinforcement learning + PyTorch Neural Network) that performs aircraft landing in super-realistic flight simulation (X-Plane 12).

In the beginning, I implemented this project in C++ from scratch, including both the RL and the neural networks, to solve many of the exercsises presented in the Barto Sutton book. The first milestone was solving a 2D tag game, the game was implemented in Java, communicating via local socket. This was later all ported to Python, including the 2D tag game and the RL code, as well as substituting the custom neural network code with PyTorch (as I realized developing neural networks from scratch is out of scope for this project and would require significant time spent just on that).

# Guide

1. Please install the xppython3 plugin before running anything: https://xppython3.readthedocs.io/en/latest/usage/installation_plugin.html

2. To have the files sync up with the xppython3's PythonPlugins directory (thats where they are executed), please edit settings.json such that it contains the correct absolute path to your PythonPlugins directory (it should work by default if you're on Windows and have installed XPLANE12 via Steam, and if you've properly installed the xppython3 plugin).

3. You have to install `torch` for this to work. You can do so by entering the game and installing it via xppython3's plugin menu.

Please use Python 3.11.7 if you want to locally install the packages so that you have an easier time working with the IDE (this is optional).