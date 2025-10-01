# PLCTestbench

PLCTestbench is a companion tool for researchers and developers working on Packet Loss Concealment (PLC). It greatly simplifies the process of measuring the reconstruction quality of PLC algorithms and allows to easily test the effects of different packet loss models and distributions. jup  It features the implementation of some of the most common packet loss models, PLC algorithms and metrics.

**Packet Loss Simulation**
- **Binomial**: uniform distribution of packet losses, governed by the Packet Error Ratio (PER) parameter.
- **Metronome**: periodic packet losses governed by the burst period, the burst length, and an offset.
- **Gilbert-Elliot**: bursty distribution of packet losses, governed by the four probabilities associated to its two states (For each state, the probability of packet loss and the probability of transitioning to the other state) [[1](#1)].

**PLC Algorithms**
- **Zeros**: the lost samples are replaced by zeros.
- **Last Packet**: the lost samples are replaced by the last received packet.
- **Low-Cost**: implementation of the algorithm proposed in [[2](#2)].
- **Burg**: Python bindings for the [C++ implementation of the Burg method](https://github.com/matteosacchetto/burg-implementation-experiments).
- **Deep Learning**: implementation of the algorithm proposed in [[3](#3)].
- **Advanced**: allows to apply different PLC algorithms to different frequency bands and audio channels (M/S processing included).
- **External**: python bindings for C++ to simplify the integration of existing algorithms.

**Metrics**
- **Mean Square Error**: the mean square error between the original and reconstructed signal.
- **Mean Amplitude Error**: the mean amplitude error between the original and reconstructed signal.
- **PEAQ**: the Perceptual evaluation of audio quality (PEAQ) metric, as defined in [[4](#4)] (**use basic mode, advanced mode needs to be verified**)
- **Windowed PEAQ**: PEAQ with a specific window length (**currenty not usable, implemantation incomplete**).
- **Spectral Energy Difference**: difference magnitude of DFT energies between the original and reconstructed signal (**currenty not usable, implemantation incomplete**).
- **Human**: this metric produces the config file for a MUSHRA test using as stimuli excerpts of the reconstructed audio tracks. It also gathers the results of the test to be displayed alongside the other metrics.
- **Perceptual**: perceptually-motivated evaluation metric for Packet loss concealment in Networked music performances, as defined in [[5](#5)]

## Installation

There is a `setup.py`. However, the functionality still needs to be verified. It is recommended to perform the following steps.
You will need a mongoDB database to store the results. You can install it locally or use a cloud service like [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
It is recomended however to use the [Docker image](https://hub.docker.com/_/mongo) provided by MongoDB.

Install WSL (https://learn.microsoft.com/de-de/windows/wsl/install).
```bash
    wsl --install
    wsl.exe -d Ubuntu
```
Choose your username and password and go to the path where you want to save your files. Create an isolated environment with Python 3.10 and activate it.
```bash
    sudo apt install python3.10 python3.10-venv python3.10-dev
    python3.10 -m venv venv310
    source venv310/bin/activate
```

Update package sources, install additional tools, add external Python package source and update again.
```bash
    sudo apt update
    sudo apt install podman
    sudo apt install software-properties-common
    sudo apt install jupyter-core
    sudo apt install libsndfile1
    sudo apt install pipx
    sudo apt install python3-pip
    sudo snap install astral-uv --classic
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
```

Install podman and pull the image.
```bash
    podman pull docker.io/library/mongo:6.0.8
```
Then run the container setting the port to 27017 and the name to mongodb. Also set the username and password for the database.
```bash
    podman run -d -p 27017:27017 --name mongodb \
    -e MONGO_INITDB_ROOT_USERNAME=myUserAdmin \
    -e MONGO_INITDB_ROOT_PASSWORD=admin \
    mongo:6.0.8
```

Clone this repository, install the requirements and the plctestbench package.
```bash
    git clone https://github.com/marcel-ellert/plc-testbench
    cd plc-testbench
    pip install -r requirements.txt
    cd ..
```

Create a folder for your input files named `input_files`.
```bash   
    mkdir input_files
```

Clone and install the [burg-python-bindings](https://github.com/LucaVignati/burg-python-bindings).
```bash
    git clone https://github.com/LucaVignati/burg-python-bindings.git
    cd burg-python-bindings
    python setup.py install
    cd ..
```

Clone and install the [cpp_plc_template](https://github.com/LucaVignati/cpp_plc_template).
```bash
    git clone https://github.com/LucaVignati/cpp_plc_template.git
    cd cpp_plc_template
    python setup.py install
    cd ..
```

Install the GSTREAMER library and the PEAQ plugin.
```bash
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio git gtk-doc-tools git2cl automake libtool
    mkdir gstpeaq && git clone https://github.com/HSU-ANT/gstpeaq.git gstpeaq
    cd gstpeaq
    aclocal && autoheader && ./autogen.sh && sed -i 's/SUBDIRS = src doc/SUBDIRS = src/' Makefile.am && ./configure --libdir=/usr/lib && automake && make && make install
    echo 'export GST_PLUGIN_PATH=/usr/lib/gstreamer-1.0:$GST_PLUGIN_PATH' >> ~/.bashrc
    source ~/.bashrc
    cd ..
```

Install webMUSHRA and pyMUSHRA.
```bash
    cd plc-testbench
    mkdir listening_tests && cd listening_tests
    mkdir db
    git clone https://github.com/audiolabs/webMUSHRA.git webmushra
    git clone https://github.com/nils-werner/pymushra.git pymushra
    pip install -e pymushra
    cd webmushra
	podman build -t webmushra -f Dockerfile .
    cd ../..
```

## Usage

You use the testbench with a Jupyter Notebook. How to use it is explained in the next steps.

### Startup and Initializing

If you have done the installation directly before, skip this point. Otherwise activate ubuntu, the virtual environment venv310 and mongodb.
```bash
    wsl -d ubuntu
    source venv310/bin/activate
    podman start mongodb
```

The file `plctestbench.ipynb` contains a Jupyter Notebook. Navigate to it and start the Juypter Notebook.
```bash
    jupyter notebook
```

Copy and paste one of the URLs to a browser. Click on `plctestbench.ipynb`. This file contains examples and explanations of how to use the tool.
Change the path to your root folder named `input_files`.
```python/jupyter notebook
    testbench_settings = {
        'root_folder': 'path/to/root/folder',
        'db_ip': 'ip.of.the.database',
        'db_port': 27017,
        'db_username': 'myUserAdmin',
        'db_password': 'admin',
    }
```
Put the audio files to be analyzed in this folder and list them as follows (path relative to `root_folder`):
```python/jupyter notebook
    original_audio_tracks = [(OriginalAudio, OriginalAudioSettings('Blues_Drums.wav')),
                            (OriginalAudio, OriginalAudioSettings('Blues_Piano.wav'))]
```
Afterwards you can set up the testbench settings by commenting in/out and/or changing the setup to your specific needs.
The format is adapted in `plctestbench.ipynb` so that simple commenting out and commenting in is possible. Here, however, it is shortened for reasons of clarity.
The basic setup looks like the following.
```python/jupyter notebook
    testbench_settings{your specifications}

    original_audio_tracks = [your specifications]
    packet_loss_simulators = [your specifications]
    plc_algorithms = [your specifications]
    metrics = [your specifications]

    testbench = PLCTestbench(original_audio_tracks, packet_loss_simulators, plc_algorithms, metrics, testbench_settings)
    testbench.run()
    testbench.plot()
```
To start the testbench you need to click on the Icon with two arrows pointing to the right.
You will find both the audio files and the results in the folder specified in `input_files`.

Important Information: The DeepLearningPLC algorithm requires the `packet_size` to be set to 128 in the `Settings` of the `PacketLossSimulator` of choice.
See the example below.
```python/jupyter notebook
    packet_loss_simulators = [(BinomialPLS, BinomialPLSSettings(packet_size = 128))]
```

### Fade in/Crossfade

When a packet is lost, the PLC algorithm has to reconstruct the lost samples. However, the reconstructed samples are not going to be identical to the original ones. A fade in/crossfade feature allows to gradually transition from the original samples to the reconstructed ones, and vice versa. This works by creating two vectors from 0 to 1 depending on the selected function. These are multiplied by the original and the reconstructed samples. Finally, the power or amplitude is adjusted.

The crossfade can be customized in the following aspects:
- **Length**: duration of the crossfade in milliseconds.
- **Function**:
    - **Power**: crossfade is $x^n$.
    - **Sinusoidal**: crossfade is sinusoidal ($\sin\left(x\right)$ for $0 < x < \frac{\pi}{2}$).
- **Exponent**: exponent of the power crossfade function.
- **Type**:
    - **Equal Power**: sum of the squares of the two crossfades equals 1.
    - **Equal Amplitude**: sum of the two crossfades equals 1.

The fade in/crossfade can be applied to the left or the right of the lost samples, or both. The following example illustrates how to configure the settings to use these.
```python/jupyter notebook
    crossfade_settings = (ManualCrossfadeSettings(length = 1, function = 'power', exponent = 1.0, type = 'power'))
    
    plc_algorithms = [(ZerosPLC, ZerosPLCSettings(fade_in = crossfade_settings, crossfade = crossfade_settings))]
```
As shown in the previous example, the `fade_in` parameter determines the crossfade to apply to the left of the lost samples, while the `crossfade` parameter determines the crossfade to apply to the right of the lost samples.

Important Information: The length of `fade_in` cannot be greater than the length of the lost packet.

The `ManualCrossfadeSettings` class allows to manually set the parameters of the crossfade.
However, there are other convenience classes that are pre-set to some common configurations.
- `NoCrossfadeSettings`: disables the crossfade.
- `LinearCrossfadeSettings`: applies a linear crossfade (`function` = 'power' and `exponent` = 1.0).
- `QuadraticCrossfadeSettings`: applies a quadratic crossfade (`function` = 'power' and `exponent` = 2.0).
- `CubicCrossfadeSettings`: applies a cubic crossfade (`function` = 'power' and `exponent` = 3.0).
- `SinusoidalCrossfadeSettings`: applies a sinusoidal crossfade (`function` = 'sinusoidal').

You will find them in `plctestbench.ipynb`.
Both the `fade_in` and `crossfade` parameters always default to `NoCrossfadeSettings` so that the crossfade is disabled by default.

### Multiband Fade in/Crossfade

Different crossfade settings can be applied to different frequency bands. This can be useful mitigate some of the artifacts introduced by the crossfade.

This is an example configuration for a quadratic crossfade of three bands without fade-in.
```python/jupyter notebook
    multiband_crossfade_settings = [
        QuadraticCrossfadeSettings(length=50),
        QuadraticCrossfadeSettings(length=5),
        QuadraticCrossfadeSettings(length=1)
    ]

    plc_algorithms = [(LastPacketPLC, LastPacketPLCSettings(
        crossfade = multiband_crossfade_settings,
        fade_in = None,
        crossfade_frequencies = [200, 2000],
        crossover_order = 4))
    ]
```
The list beginning with the `multiband_crossfade_settings` class contains the crossfade settings for each band. The first element of the list is the crossfade settings for the first band, the second element is the crossfade settings for the second band, and so on. The length of the list must be equal to the number of bands. Each band can have its own crossfade settings, totally unrelated to the other bands. The `crossfade_frequencies` class allows to specify the frequencies of the bands. The first frequency is the upper bound of the first band, while the last frequency is the lower bound of the last band. The number of frequencies determines the number of bands. The `crossover_order` describes the slope of the filters of the frequency bands. 2 means 12 dB/octave, 4 means 24 dB/octave and 8 means 48 dB/octave.

### Advanced PLC

The `AdvancedPLC` object allows for two complex behaviours simultaneously:
- **Multiband PLC**: different PLC algorithms can be applied to different frequency bands. 
- **Spatial PLC**: different PLC algorithms can be applied to different audio channels (L/R: left/right or M/S: mid/side).

An example configuration for Multiband PLC is the following.
```python/jupyter notebook
    plc_algorithms = [(AdvancedPLC, AdvancedPLCSettings(
        settings = {'linked':[ZerosPLCSettings(),
                            LastPacketPLCSettings(),
                            BurgPLCSettings()]},
        frequencies = {'linked': [200, 2000]},
        channel_link=True))
    ]
```
The `frequencies` parameter is a dictionary that maps the name of the channel to a list of frequencies. The `settings` parameter is a dictionary that maps the name of the channel to a list of tuples. Each tuple contains the PLC algorithm to apply to the band and its settings. The length of each list must be equal to the number of bands of that channel.
If L/R or M/S channels require different PLC algorithms, the `channel_link` parameter needs to be set to `False` and the related settings need to be specified in the `settings` parameter using the `left` and `right` or `mid` and `side` keys. For example configurations take a look into `plctestbench.ipynb`.

## References
    
<a id="1">[1]</a>
Elliott, Edwin O. "Estimates of error rates for codes on burst-noise channels." The Bell System Technical Journal 42.5 (1963): 1977-1997.

<a id="2">[2]</a>
Fink, Marco, and Udo Zölzer. "Low-Delay Error Concealment with Low Computational Overhead for Audio over IP Applications." DAFx. 2014.
    
<a id="3">[3]</a> 
Verma, Prateek, et al. "A deep learning approach for low-latency packet loss concealment of audio signals in networked music performance applications." 2020 27th Conference of open innovations association (FRUCT). IEEE, 2020.
    
<a id="4">[4]</a> 
Thiede, Thilo, et al. "PEAQ-The ITU standard for objective measurement of perceived audio quality." Journal of the Audio Engineering Society 48.1/2 (2000): 3-29.

<a id="5">[5]</a> 
Vignati, Luca, and Luca Turchet, "On the lack of a perceptually-motivated evaluation metric for Packet Loss Concealment in Networked Music Performances." Journal of the Audio Engineering Society, 2025.