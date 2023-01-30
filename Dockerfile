FROM hardikparwana/cuda116desktop:v1
RUN apt-get update
RUN apt install -y python3-pip
RUN pip3 install numpy==1.22.3 matplotlib sympy argparse scipy==1.8.0
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
