FROM nvidia/cuda:8.0-devel-ubuntu16.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

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

# install director
COPY ./install_director.sh /tmp/install_director.sh
RUN yes "Y" | /tmp/install_director.sh

# set the terminator inside the docker container to be a different color
RUN mkdir -p .config/terminator
COPY ./terminator_config .config/terminator/config
RUN chown $USER_NAME:$USER_NAME -R .config


# change ownership of everything to our user
RUN cd $WORKDIR && chown $USER_NAME:$USER_NAME -R .



ENTRYPOINT bash -c "source ~/code/docker/entrypoint.sh && /bin/bash"


