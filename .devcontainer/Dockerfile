FROM tiangolo/python-machine-learning:python3.6-tensorflow

RUN conda install pandas numpy matplotlib jupyter notebook
RUN conda install -c pytorch pytorch 
RUN conda install -c anaconda opencv

EXPOSE 8888

ARG UNAME=dlvr
ARG WS_DIR=/home/$UNAME/code
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -s /bin/bash -N -u $UID $UNAME && \
    adduser $UNAME sudo && \
    mkdir -p $WS_DIR/ &&\
    chown $UNAME:$GID -R $WS_DIR

USER $UNAME
WORKDIR $WS_DIR/

RUN echo "alias l=\"ls -alhG\"" >> ~/.bashrc
RUN echo "alias ..=\"cd ..\"" >> ~/.bashrc
RUN echo "alias gst=\"git status\""  >> ~/.bashrc
RUN echo "alias gps=\"git push\""  >> ~/.bashrc
RUN echo "alias gpl=\"git pull\""  >> ~/.bashrc
RUN echo "alias gcm=\"git commit -m\""  >> ~/.bashrc
RUN echo "alias gad=\"git add\""  >> ~/.bashrc
RUN echo "alias gco=\"git checkout\""  >> ~/.bashrc
RUN echo "alias gdf=\"git diff\""  >> ~/.bashrc
RUN echo "alias gup=\"git add . && git commit -m 'update' && git push\""  >> ~/.bashrc
RUN echo "alias code=\cd /home/dlvr/code\""

CMD cd ${WS_DIR} && jupyter notebook --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' && /bin/bash
