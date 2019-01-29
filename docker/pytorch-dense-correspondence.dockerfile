# FROM nvidia/cuda:9.0-devel-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

# see http://gbraad.nl/blog/non-root-user-inside-a-docker-container.html
# RUN dnf install -y sudo && \
#     adduser $USER_NAMEuser && \
#     echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
#     chmod 0440 /etc/sudoers.d/user



RUN apt-get update
RUN apt install sudo
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME 
RUN groupmod -g $USER_GID $USER_NAME

WORKDIR /home/$USER_NAME
# require no sudo pw in docker
# RUN echo $USER_PASSWORD | sudo -S bash -c 'echo "'$USER_NAME' ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/docker-user' && printf "\n"

COPY ./install_dependencies.sh /tmp/install_dependencies.sh
RUN yes "Y" | /tmp/install_dependencies.sh

COPY ./install_pytorch.sh /tmp/install_pytorch.sh
RUN yes "Y" | /tmp/install_pytorch.sh

COPY ./install_more.sh /tmp/install_more.sh
RUN yes "Y" | /tmp/install_more.sh

COPY ./install_poser.sh /tmp/install_poser.sh
RUN yes "Y" | /tmp/install_poser.sh

# install director
COPY ./install_director.sh /tmp/install_director.sh
RUN yes "Y" | /tmp/install_director.sh





# set the terminator inside the docker container to be a different color
RUN mkdir -p .config/terminator
COPY ./terminator_config .config/terminator/config
RUN chown $USER_NAME:$USER_NAME -R .config

# install GLX-Gears
RUN apt-get update && apt-get install -y \
   mesa-utils && \
   rm -rf /var/lib/apt/lists/*



# needed to get OpenGL running inside the docker
# https://github.com/machinekoder/nvidia-opengl-docker

# optional, if the default user is not "root", you might need to switch to root here and at the end of the script to the original user again.
# e.g.
# USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libxau-dev \
        libxdmcp-dev \
        libxcb1-dev \
        libxext-dev \
        libx11-dev && \
    rm -rf /var/lib/apt/lists/*

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
# e.g. nvidia/opengl:1.1-glvnd-runtime-ubuntu16.04)
COPY --from=machinekoder/nvidia-opengl-docker:1.1-glvnd-runtime-stretch \
  /usr/local/lib/x86_64-linux-gnu \
  /usr/local/lib/x86_64-linux-gnu

# replace with other Ubuntu version if desired
# see: https://hub.docker.com/r/nvidia/opengl/
# e.g. nvidia/opengl:1.1-glvnd-runtime-ubuntu16.04
COPY --from=machinekoder/nvidia-opengl-docker:1.1-glvnd-runtime-stretch \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json \
  /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
    ldconfig && \
    echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
    echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# USER original_user

COPY ./install_coco_api.sh /tmp/install_coco_api.sh
RUN yes "Y" | /tmp/install_coco_api.sh


# make python3 pytorch 1.0 virtualenv
RUN cd $WORKDIR
ENV PYTHON3_PYTORCH_VIRTUALENV_DIR=/home/${USER_NAME}/venv_python3
COPY ./install_python3_pytorch_virtualenv.sh /tmp/install_python3_pytorch_virtualenv.sh
RUN yes "Y" | /tmp/install_python3_pytorch_virtualenv.sh

# change ownership of everything to our user
RUN cd $WORKDIR
RUN chown $USER_NAME:$USER_NAME -R .



ENTRYPOINT bash -c "source ~/code/docker/entrypoint.sh && /bin/bash"


