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
from elaas.family.Deep_128f_3c import Deep_128f_3c
from elaas.family.Deep_100f import Deep_100f
from elaas.family.Deep_160f_2c import Deep_160f_2c


import paho.mqtt.client as mqtt
import time
import timeit

_g_cst_ToMQTTTopicServerIP = "localhost"
_g_cst_ToMQTTTopicServerPort = 1883 #port
_g_cst_MQTTTopicName = "INFERENCE_ANS" #TOPIC name

model_1 = Cifar10_4f3l()
model_2 = Cifar10_64f4l()
model_3 = Cifar10_64f()
model_4 = Cifar10_64f_nb()
model_5 = Cifar10_64f_nb_good()
model_6 = Cifar10_64f_nb_good2()
model_7 = Cifar10_32f_nb_good2_75()
model_8 = Cifar10_16f_nb_good2_75()
model_9 = Deep_100f()
model_10 = Deep_128f_3c()
model_11 = Deep_160f_2c()

def on_connect(client, userdata, flags, rc):
    print("Connected wit result code "+str(rc))

    client.subscribe("INFERENCE_REQ")

def on_message(client, userdata, msg):
    #print(msg.topic+" "+str(msg.payload))
    #message = str(msg.payload)
    tStart = time.time()

    time.sleep(0.02)

    x = list(bytearray(msg.payload))
    #print(type(x))
    #for i in range(100):
    #    print x[i]
    model = x[0]
    data = x[1:]
    #print data
    if(model == 1):
        a , eva_T = model_1(data)
    elif(model == 2):
        a , eva_T = model_2(data)
    elif(model == 3):
        a , eva_T = model_3(data)
    elif(model == 4):
        a , eva_T = model_4(data)
    elif(model == 5):
        a , eva_T = model_5(data)
    elif(model == 6):
        a , eva_T = model_6(data)
    elif(model == 7):
        a , eva_T = model_7(data)
    elif(model == 8):
        a , eva_T = model_8(data)
    elif(model == 9):
        a , eva_T = model_9(data)
    elif(model == 10):
        a , eva_T = model_10(data)
    elif(model == 11):
        a , eva_T = model_11(data)
    else:
        a = "-1"

    tEnd = (time.time())
    server_CT = tEnd - tStart

    a = str(a) + ":" + str(server_CT)[:8] + ":" + str(eva_T)[:8] + ":\0"
    print 'result = %s' % (a)
    client.publish(_g_cst_MQTTTopicName, a)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 60)

client.loop_forever()
