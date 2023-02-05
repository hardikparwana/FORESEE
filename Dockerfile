FROM hardikparwana/cuda116desktop:v1
RUN apt-get update
RUN apt install -y python3-pip
RUN pip3 install numpy==1.22.3 matplotlib sympy argparse scipy==1.8.0
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN apt-get -y install vim
RUN pip3 install cvxpy cvxpylayers gpytorch
RUN apt -y install ffmpeg
RUN pip3 install gym==0.26.0 gym-notices==0.0.8 gym-recording==0.0.1 moviepy==1.0.3 pygame==2.1.2
