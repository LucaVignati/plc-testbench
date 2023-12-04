# Install react frontend
FROM node:18.15.0-slim as ui-frontend-build
RUN apt-get update
RUN apt-get install git --yes --force-yes
WORKDIR /plc-testbench-ui
RUN git clone https://github.com/stefano-dallona/react-test.git
WORKDIR /plc-testbench-ui/react-test
RUN npm config set strict-ssl false
RUN npm install --force
RUN cp .env.docker .env.local
RUN npm run build
# && mkdir frontend && cp build/* /plc-testbench-ui/frontend

FROM python:3.8-slim-buster as ui-backend-build
#FROM tiangolo/uwsgi-nginx-flask:python3.8-alpine

WORKDIR /plc-testbench-ui

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements.txt .

RUN apt-get update
RUN apt-get install libsndfile1-dev --yes --force-yes
# Install gstremer dependencies
RUN apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio git gtk-doc-tools git2cl --yes --force-yes
# Start of download and compile gstpeaq
RUN mkdir gstpeaq && git clone https://github.com/HSU-ANT/gstpeaq.git gstpeaq
WORKDIR /plc-testbench-ui/gstpeaq
RUN aclocal && autoheader && ./autogen.sh && sed -i 's/SUBDIRS = src doc/SUBDIRS = src/' Makefile.am && ./configure --libdir=/usr/lib && automake && make && make install
# End of download and compile gstpeaq
# Install ui dependencies
WORKDIR /plc-testbench-ui
COPY requirements.txt /tmp
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r /tmp/requirements.txt
# Install burg-python-bindings
WORKDIR /
RUN git clone https://github.com/LucaVignati/burg-python-bindings.git && cd burg-python-bindings && python setup.py install
# Install cpp_plc_template
WORKDIR /
RUN git clone https://github.com/LucaVignati/cpp_plc_template.git && cd cpp_plc_template && python setup.py install
# Install plctestbench
WORKDIR /plc-testbench
COPY . .
RUN cd /plc-testbench && python setup.py sdist && python3 -m pip install -f ./dist plc-testbench && cp -r dl_models /plc-testbench-ui

WORKDIR /plc-testbench-ui
RUN git clone https://github.com/stefano-dallona/plc-testbench-ui.git

COPY --from=ui-frontend-build /plc-testbench-ui/react-test/build /plc-testbench-ui/frontend/build/

ENTRYPOINT [ "python3", "app.py" ]