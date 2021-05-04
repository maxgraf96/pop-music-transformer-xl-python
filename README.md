# Computational Creativity Submission: Readme and Important Notes
Welcome! This file includes everything you need to know to get the code running. 
There are two main components: 

 - Python backend (including the generative model)
 - Node.JS server (serving the frontend and communicating with backend)

They interact with each other as indicated here:
![High level view of the system](https://i.ibb.co/kxwtQqT/hlo.png)
 
 The first and only thing to know is that it is: ***Always start the Node.JS server  before the python backend!***

# Prerequisites
There are a few prerequisites to running the system. 
The backend requires **Python >=3.6** and a virtual environment (https://docs.python.org/3/tutorial/venv.html).
 
 For the frontend **Node.JS** (https://nodejs.org/en/) is required. The system was tested with version 14, but any version >= 14 should work.

# Setup
Once you have Python and Node.JS installed the steps are as follows:
*These steps assume that the main submission folder is called 'submission'.*
**Node.JS server**:
- Open a terminal
- Navigate to the main submission folder
- Execute `cd CC_Project_Frontend`
- Execute `npm i` to install the required Node.JS packages
- Execute `npm run start` to start the Node.JS server
- Proceed with starting the python backend

**Python backend**:
Step 1: Get the pre-trained model
- Download the pre-trained model from https://drive.google.com/file/d/1oX0ggfZiGRePpye6M-rwdPt2msAuei0M/view?usp=sharing
- Place the model.pt file in the `submission/PopMusicTransformerPytorch/src/transformer/REMI-my-checkpoint` folder.

Step 2: Create python environment
- Create and activate a new virtual python environment (see https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments)
- Open a terminal
- Navigate to the main submission folder
- Execute `cd PopMusicTransformerPytorch`
- If your python version is **3.6.x**: Execute `pip install -r requirements_python_3_6.txt`
- If your python version is **>3.6**: Execute `requirements_python_3_7+.txt`
- Execute `cd src/transformer`
- Execute `python main.py` or `python3 main.py`, depending on your bash aliases

Once both the Node.JS server and the Python backend are running, open a web-browser and go to [http://localhost:5000/](http://localhost:5000/).
Click Generate to generate some MIDI from scratch using the pre-trained model!

# Frontend controls

 1. Click "Generate" to generate some MIDI from scratch - please be patient with the MIDI generation, it can take a few seconds depending on your CPU/GPU
 2. Hit 1, 2 or 3 to select and play track 1, 2 or 3
 3. Clicking a track selects it, but doesn't play it back
 4. After selecting a track it will be highlighted with a large blurry green border! Hit **ENTER** to generate new MIDI based on the selected track
 5. Go back to step 2 (or 1 if you wanna start from scratch)
 6. Repeat until happy
 7. The outputs are saved in `CC_Project_Frontend/midi` after each generation.
