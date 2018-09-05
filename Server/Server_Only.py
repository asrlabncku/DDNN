# SubscriberTest.py
import os
import sys
sys.path.append('..')
import chainer
from elaas.family.cifar10_model_4f3l import Cifar10_4f3l
from elaas.family.cifar10_model_64f4l import Cifar10_64f4l
from elaas.family.cifar10_model_64f import Cifar10_64f
from elaas.family.cifar10_model_64f_nb import Cifar10_64f_nb
from elaas.family.cifar10_model_64f_nb_good import Cifar10_64f_nb_good
from elaas.family.cifar10_model_64f_nb_good2 import Cifar10_64f_nb_good2
from elaas.family.cifar10_model_32f_nb_good2_75 import Cifar10_32f_nb_good2_75
from elaas.family.cifar10_model_16f_nb_good2_75 import Cifar10_16f_nb_good2_75

import paho.mqtt.client as mqtt
import time
import struct

_g_cst_ToMQTTTopicServerIP = "localhost"
_g_cst_ToMQTTTopicServerPort = 1883 #port
_g_cst_MQTTTopicName = "INFERENCE_RAW_ANS" #TOPIC name

model_1 = Cifar10_4f3l()
model_2 = Cifar10_64f4l()
model_3 = Cifar10_64f()
model_4 = Cifar10_64f_nb()
model_5 = Cifar10_64f_nb_good()
model_6 = Cifar10_64f_nb_good2()
model_7 = Cifar10_32f_nb_good2_75()
model_8 = Cifar10_16f_nb_good2_75()

def on_connect(client, userdata, flags, rc):
    print("Connected wit result code "+str(rc))
    client.subscribe("INFERENCE_RAW_REQ")

def on_message(client, userdata, msg):
    #print(msg.topic+" "+str(msg.payload))
    #message = str(msg.payload)
    #print "message = %s" % message
    tStart = time.time()
        
    #print len(msg.payload)
    time.sleep(0.02)    
    f_num = ""
    for i in range(len(msg.payload)/4):
        f_num = f_num + "f"
    x = list(struct.unpack(f_num , msg.payload))

    model = x[0]
    data = x[1:]
    
    if(model == 1):
        a , eva_Ts = model_1(data)
    elif(model == 2):
        a , eva_Ts = model_2(data)
    elif(model == 3):
        a , eva_Ts = model_3(data)
    elif(model == 4):
        a , eva_Ts = model_4(data)
    elif(model == 5):
        a , eva_Ts = model_5(data)
    elif(model == 6):
        a , eva_Ts = model_6(data)
    elif(model == 7):
        a , eva_Ts = model_7(data)
    elif(model == 8):
        a , eva_Ts = model_8(data)
    else:
        a = "-1"

    tEnd = time.time()
    server_CT = tEnd - tStart
    a = str(a) + ":" + str(server_CT) + ":" + str(eva_Ts) + ":"
    print 'result = %s' % (a)
    client.publish(_g_cst_MQTTTopicName, a)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 60)

client.loop_forever()
