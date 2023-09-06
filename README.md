# PLCTestbench

PLCTestbench is a companion tool for researchers and developers working on Packet Loss Concealment (PLC). It greatly simplifies the process of measuring the reconstruction quality of PLC algorithms and allows to easily test the effects of different packet loss models and distributions.

It features the implementation of some of the most common packet loss models, PLC algorithms and metrics:

**Packet Loss Simulation**
- **Binomial**: uniform distribution of packet losses, governed by the Packet Error Ratio (PER) parameter.
- **Gilbert-Elliot**: bursty distribution of packet losses, governed by the four probabilities associated to its two states (For each state, the probability of packet loss and the probability of transitioning to the other state) [[1](#1)].

**PLC Algorithms**
- **Zeros**: the lost samples are replaced by zeros.
- **Last Packet**: the lost samples are replaced by the last received packet.
- **Low-Cost**: implementation of the algorithm proposed in [[2](#2)].
- **Burg**: Python bindings for the [C++ implementation of the Burg method](https://github.com/matteosacchetto/burg-implementation-experiments).
- **Deep Learning**: implementation of the algorithm proposed in [[3](#3)].
- **External**: Python bindings for C++ to simplify the integration of existing algorithms.

**Metrics**
- **Mean Square Error**: the mean square error between the original and reconstructed signal.
- **PEAQ**: the Perceptual Evaluation of Audio Quality (PEAQ) metric, as defined in [[4](#4)].
## Basic Usage

You will need a mongoDB database to store the results. You can install it locally or use a cloud service like [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
It is recomended however to use the [Docker image](https://hub.docker.com/_/mongo) provided by MongoDB.

Pull the image
```bash
    docker pull mongo:6.0.8
```
Then run the container setting the port to 27017 and the name to mongodb. Also set the username and password for the database.
```bash
    docker run -d -p 27017:27017 --name mongodb \
    -e MONGO_INITDB_ROOT_USERNAME=myUserAdmin \
    -e MONGO_INITDB_ROOT_PASSWORD=admin \
    mongo:6.0.8
```

Clone this repository, install the requirements and the plctestbench package:

```bash
    git clone https://github.com/LucaVignati/plc-testbench.git
    cd plc-testbench
    pip install -r requirements.txt
    pip install .
```

Clone and install the [burg-python-bindings](https://github.com/LucaVignati/burg-python-bindings):
```bash
    git clone https://github.com/LucaVignati/burg-python-bindings.git
    cd burg-python-bindings
    python setup.py install
```

Clone and install the [cpp_plc_template](https://github.com/LucaVignati/cpp_plc_template):
```bash
    git clone https://github.com/LucaVignati/cpp_plc_template.git
    cd cpp_plc_template
    python setup.py install
```

The file `plctestbench.ipynb` contains a Jupyter Notebook with a basic example of how to use the tool.

Input the settings of the testbench as follows:
```python
    testbench_settings = {
        'root_folder': 'path/to/root/folder',
        'db_ip': 'ip.of.the.database',
        'db_port': 27017,
        'db_username': 'myUserAdmin',
        'db_password': 'admin',
}
```

List the audio files you want to input as follows (path relative to `root_folder`):
```python
original_audio_tracks = [(OriginalAudio, OriginalAudioSettings('Blues_Drums.wav')),
                         (OriginalAudio, OriginalAudioSettings('Blues_Piano.wav'))]
```

List the packet loss models you want to test as follows:
```python
packet_loss_simulators = [(GilbertElliotPLS, GilbertElliotPLSSettings()),
                          (BinomialPLS, BinomialPLSSettings())]
```

List the PLC algorithms you want to test as follows:
```python
plc_algorithms = [(ZerosPLC, ZerosPLCSettings()),
                  (LastPacketPLC, LastPacketPLCSettings()),
                  (LowCostPLC, LowCostPLCSettings()),
                  (BurgPLC, BurgPLCSettings()),
                  (DeepLearningPLC, DeepLearningPLCSettings()),
                  (ExternalPLC, ExternalPLCSettings())]
```
❗The DeepLearningPLC algorithm requires the `bufer_size` to be set to 128 in the `Settings` of the `PacketLossSimulator` of choice.

List the metrics you want to use as follows:
```python
metrics = [(MSECalculator, MSECalculatorSettings()),
           (PEAQCalculator, PEAQCalculatorSettings())]
```

Finally, run the testbench:
```python
testbench = Testbench(testbench_settings, user, original_audio_tracks, packet_loss_simulators, plc_algorithms, metrics)
testbench.run()
```

If you want to change the parameters of any of the modules, you can do so by passing the settings as a parameter to the constructor of the module. For example, to change the PER of the Binomial packet loss model:
```python
packet_loss_simulators = [(BinomialPLS, BinomialPLSSettings(per=0.1))]
```

For the full list of settings, check the `settings.py` file.

To plot the results:
```python
testbench.plot(to_file=True, original_tracks=True, lost_samples_masks=True, output_analyses=True)
```

You can also plot the waveform of the reconstructed audio tracks, however since we're plotting the entire duration of the audio file, the differences with the original tracks are not going to be visible. This is why we developed a user interface for this application.

You will find both the audio files and the results in the folder specified in the `root_folder` setting.

## User Interface
This user interface is an ongoing thesis project carried out by Stefano Dallona under the supervision of Luca Vignati.
It is a web application developed using the React framework.
The code is available in the following two repositories:

- [plc-testbench-ui](https://github.com/stefano-dallona/plc-testbench-ui)
- [react-test](https://github.com/stefano-dallona/react-test)

The easiest way to try it out is to use the Docker image provided by Stefano Dallona:
```bash
    docker pull stdallona/plc-testbench-ui:1.0.1
```
This Docker image already contains the code of PLCTestbench so it only requires a running MongoDB instance (see previous section).

Run the following command to start the container:
```bash
    docker run -e DB_USERNAME=$DB_USERNAME \
               -e DB_PASSWORD=$DB_PASSWORD \
               -e DB_HOST=$DB_HOST \
               -e DB_CONN_STRING=$DB_CONN_STRING \
               -e GEVENT_SUPPORT=$GEVENT_SUPPORT \
               -e FLASK_APP=$FLASK_APP \
               -e FLASK_DEBUG=$FLASK_DEBUG \
               -e FRONTEND_DATA_FOLDER=$FRONTEND_DATA_FOLDER \
               -e SECURITY_ENABLED=$SECURITY_ENABLED \
               -p 5000:5000 \
               -v /path/to/root/folder:/original_tracks \
               --name plc-testbench-ui \
               stdallona/plc-testbench-ui:1.0.1
```
Where the environment variables are:
| Variable | Value | Description |
| --- | --- | --- |
| DB_USERNAME | myUserAdmin | Username of the database |
| DB_PASSWORD | admin | Password of the database |
| DB_HOST | ip.of.the.database | IP address of the database |
| DB_CONN_STRING | mongodb://ip:27017 | Connection string of the database |
| GEVENT_SUPPORT | True | Enable gevent support |
| FLASK_APP | app.py | Flask application |
| FLASK_DEBUG | True | Enable Flask debug mode |
| FRONTEND_DATA_FOLDER | /original_tracks | Path to the folder containing the audio files |
| SECURITY_ENABLED | False | Enable security |

Then open your browser and go to `localhost:5000`.

❗Please consider the pre-release status of this user interface when using it.

## References
    
<a id="1">[1]</a>
Elliott, Edwin O. "Estimates of error rates for codes on burst-noise channels." The Bell System Technical Journal 42.5 (1963): 1977-1997.

<a id="2">[2]</a>
Fink, Marco, and Udo Zölzer. "Low-Delay Error Concealment with Low Computational Overhead for Audio over IP Applications." DAFx. 2014.
    
<a id="3">[3]</a> 
Verma, Prateek, et al. "A deep learning approach for low-latency packet loss concealment of audio signals in networked music performance applications." 2020 27th Conference of open innovations association (FRUCT). IEEE, 2020.
    
<a id="4">[4]</a> 
Thiede, Thilo, et al. "PEAQ-The ITU standard for objective measurement of perceived audio quality." Journal of the Audio Engineering Society 48.1/2 (2000): 3-29.
