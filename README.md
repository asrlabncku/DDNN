# DDNN Adaptive Framework Deployment

## 1. Environment Setup

### Install DDNN environment by following link:
https://hackmd.io/kN4lb_NCQK-WKO0Rkp8tbg
(Install to the Chainer installing step) 

### Install MQTT Service 
* Server : 
    * broker (python)
    ```shell=
    sudo apt-get install mosquitto mosquitto-clients
    ```
    * client (python)
    ```shell=
    sudo pip install paho-mqtt
    ```
    
* End : 
    * client (C code)
    ```shell=
    sudo apt-get install open-ssl libssl-dev
    git clone https://github.com/eclipse/paho.mqtt.c.git
    cd org.eclipse.paho.mqtt.c.git
    sudo make
    sudo make install
    ```

### Basic requirements: 
* **Server (cloud):**
    * Ubuntu 16.04 or higher version
    * Python 2.7
    * Nvidia vendor GPU or accelerator  
* **Device (end or edge):**
    * Ubuntu 16.04 or higher
    * GCC (linux 5.4.0 or higher)
    * (OpenCL 1.2 or 2.0)

## 2.Download source code
Download the source code 
https://drive.google.com/file/d/18q-5_wKKEEDTA4T29CXvDMouxnT-rpPF/view?usp=sharing

## 3.Deployment 	
The deployment has two part, Device and Server:

* ### **Server Deployment:** 
    * Setup DDNN framework into server
    ```shell=
    cd Cloud_device
    cd Framework
    sudo ./building.sh
    ```
* ### **Device Deployment:** 
    * Compile the C program which is generated.
    ```shell=
    cd End_device
    vim mqtt_async_t.h  //Change borker of server IP by '#define ADDRESS "tcp://xxx.xxx.xxx.xxx:1883"'
    sudo ./building.sh
    gcc Source.c -o output -lm -lpaho-mqtt3a
    ```
## 4.Execution
First execute Cloud and then execute the End device.
* ### **Server Execution** 
    Run the broker at first
    ```shell=
    cd Cloud_device
    python broker_develop.py
    ```
    
* ### **End Execution** 
    Run the inference request after broker is executed on server
    ```shell=
    cd End_device
    sudo chmod +x cifar10*
    ./cifar10_oneinput.sh
    ```
