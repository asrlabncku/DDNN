#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>
#include "MQTTClient.h"

#pragma warning( disable : 4996 )

float Server_CT = 0.0;
float Server_ET = 0.0;
float Pass_T = 0.0;
int Final_ep = 0;
//struct timeval t0,t2;

#define ADDRESS     "tcp://140.116.245.173:1883"
#define CLIENTID    "ExampleClientPub"
#define TOPIC_REQ   "INFERENCE_RAW_REQ"
#define TOPIC_REC   "INFERENCE_RAW_ANS"
#define QOS         1
#define TIMEOUT     10000L

//===========================MQTT Global==============================
int msgarrvd_t = 0;
volatile MQTTClient_deliveryToken deliveredtoken;
struct timeval c_t0,c_t1;

MQTTClient client;
MQTTClient_connectOptions conn_opts;
MQTTClient_message pubmsg;
MQTTClient_deliveryToken token;
int rc;

char *mes[4] = {""};
int payload_size;
unsigned char* payload_g;

//=============================================== Function ========================================

void split(char **arr, char *str, const char *del) { // Split the token from the input textfile to char* array.
        char *s = strtok(str, del);
        int i = 0;
        while (s != NULL) {
                arr[i] = s;
                i++;
                s = strtok(NULL, del);
        }
}

void delivered(void *context, MQTTClient_deliveryToken dt)
{
//    printf("Message with token value %d delivery confirmed\n", dt);
    deliveredtoken = dt;
}

void connlost(void *context, char *cause)
{
    printf("\nConnection lost\n");
    printf("     cause: %s\n", cause);
}

int msgarrvd(void *context, char *topicName, int topicLen, MQTTClient_message *message)
{
    int i;
    char* payloadptr;
    gettimeofday(&c_t1, NULL);
    payloadptr = message->payload;

    split(mes, payloadptr, ":");

    MQTTClient_freeMessage(&message);
    MQTTClient_free(topicName);
    msgarrvd_t = 1;
    return 1;
}

//=======================================Service Setup======================================================

void MQTT_initialization()
{    
    MQTTClient_connectOptions conn_opts_o = MQTTClient_connectOptions_initializer;
    conn_opts = conn_opts_o;
    MQTTClient_message pubmsg_o = MQTTClient_message_initializer;
    pubmsg = pubmsg_o;
    MQTTClient_create(&client, ADDRESS, CLIENTID, MQTTCLIENT_PERSISTENCE_NONE, NULL);
    conn_opts.keepAliveInterval = 20;
    conn_opts.cleansession = 1;
    MQTTClient_setCallbacks(client, NULL, connlost, msgarrvd, delivered);
    if ((rc = MQTTClient_connect(client, &conn_opts)) != MQTTCLIENT_SUCCESS)
    {
        printf("Failed to connect, return code %d\n", rc);
        exit(EXIT_FAILURE);
    }
}

// ======================================Entropy Compute=======================================

void softmax(float *input, int input_len)
{
	assert(input != NULL);
	assert(input_len != 0);
	int i;
	float m;
	/* Find maximum value from input array */
	m = input[0];
	for (i = 1; i < input_len; i++) {
		if (input[i] > m) {
			m = input[i];
		}
	}
	float sum = 0;
	for (i = 0; i < input_len; i++) {
		sum += expf(input[i] - m);
	}
	for (i = 0; i < input_len; i++) {
		input[i] = expf(input[i] - m - log(sum));

	}
}

float entropy(float *x, int C)
{
	float result = 0, sum_x = 0;
	softmax(x, C);
	for (int i = 0; i < C; i++)
	{
		x[i] = log(x[i]) * x[i];
		sum_x = sum_x += x[i];
	}
	result = -(sum_x / log(abs(C)));
	return result;
}

//====================================================Inference Request===============================================

int pass_size_raw(int n , int f , int w , int h)
{
	return (6 + n*f*w*h);
}

void conn_server( float* payload, int size){
    pubmsg.payload = payload;
    pubmsg.payloadlen = size;
    pubmsg.qos = QOS;
    pubmsg.retained = 0;
    deliveredtoken = 0;
    msgarrvd_t = 0;

    gettimeofday(&c_t0, NULL);
    MQTTClient_publishMessage(client, TOPIC_REQ, &pubmsg, &token);
    MQTTClient_subscribe(client, TOPIC_REC, QOS);
    do{
    }while(msgarrvd_t == 0);

}

void pass_cloud_raw(int model, float* x, int layer_number, int m, int d, int w, int h){
        int data_size;		
	data_size = pass_size_raw(m, d, w, h);

	float * data = malloc(data_size * sizeof(float));

        data[0] = (float)model;
	data[1] = (float)layer_number;
	data[2] = (float)m;
	data[3] = (float)d;
	data[4] = (float)w;
	data[5] = (float)h;
        
        for(int i = 0 ; i < m*d*w*h ; i++)
        	data[6 + i] = x[i];
        
	//gettimeofday(&t0, NULL);
        conn_server(data , data_size*sizeof(float));
        //gettimeofday(&t2, NULL);
        
        Server_CT = atof(mes[1]);
        Server_ET = atof(mes[2]);
        Final_ep = atoi(mes[0]);
	free(data);
}


