#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>
#include "MQTTClient.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)

#pragma warning( disable : 4996 )

float Server_CT = 0.0;
float Server_ET = 0.0;
float Pass_T = 0.0;
int Final_ep = 0;
//struct timeval t0,t2;

#define ADDRESS     "tcp://140.116.245.173:1883"
#define CLIENTID    "ExampleClientPub"
#define TOPIC_REQ   "INFERENCE_REQ"
#define TOPIC_REC   "INFERENCE_ANS"
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

//==========================OpenCL Global=============================
FILE *fp;
char *source_str;
size_t source_size;
char *source_str2;
size_t source_size2;

cl_platform_id platform_ids[10];
cl_device_id device_ids[10];
cl_device_id device_sel = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
char vendor[1024];
char device_name[1024];
cl_context context_init;
cl_command_queue command_queue;

cl_kernel kernelf;
cl_kernel kernelb;

cl_program programf;
cl_program programb;

cl_int ret;

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

void OpenCL_initialization(){
        fp = fopen("conv_pool_f_op1.cl", "r");
        if (!fp) {

                fprintf(stderr, "Failed to load kernel.\n");
                //system("pause");
                exit(1);
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        fp = fopen("conv_pool_b_op1.cl", "r");
        if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                //system("pause");
                exit(1);
        }
        source_str2 = (char*)malloc(MAX_SOURCE_SIZE);
        source_size2 = fread(source_str2, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        ret = clGetPlatformIDs(10, platform_ids, &ret_num_platforms);
        //printf("Platform number : %u \n", ret_num_platforms);

        for (int i = 0; i < ret_num_platforms; i++)
        {
                clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
                //printf("Platform [%d] : %s \n", i, vendor);
                ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_DEFAULT, 1,
                        device_ids, &ret_num_devices);

                //printf("Device number : %u\n", ret_num_devices);

                for (int j = 0; j < ret_num_devices; j++)
                {
                        clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
                        //printf("Device [%d] : %s \n", j, device_name);
                        device_sel = device_ids[j];
                }
        }
        //printf("Get device state : %d \n", ret);

        context_init = clCreateContext(NULL, 1, &device_sel, NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context_init, device_sel, 0, &ret);
        //printf("Command queue state : %d \n", ret);

        programf = clCreateProgramWithSource(context_init, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
        ret = clBuildProgram(programf, 1, &device_sel, NULL, NULL, NULL);

        // Shows the log
        if (ret != CL_SUCCESS)
        {
                char* build_log;
                size_t log_size;
                clGetProgramBuildInfo(programf, device_sel, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                build_log = (char*)malloc(sizeof(char)*(log_size + 1));
                clGetProgramBuildInfo(programf, device_sel, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
                build_log[log_size] = '\0';
                printf(build_log, "\n");
                free(build_log);
        }

        programb = clCreateProgramWithSource(context_init, 1, (const char **)&source_str2, (const size_t *)&source_size2, &ret);
        ret = clBuildProgram(programb, 1, &device_sel, NULL, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
                char* build_log;
                size_t log_size;
                clGetProgramBuildInfo(programb, device_sel, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                build_log = (char*)malloc(sizeof(char)*(log_size + 1));
                clGetProgramBuildInfo(programb, device_sel, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
                build_log[log_size] = '\0';
                printf(build_log, "\n");
                free(build_log);
        }

        kernelf = clCreateKernel(programf, "conv_pool_f", &ret);
        kernelb = clCreateKernel(programb, "conv_pool_b", &ret);
        //printf("======================= OpenCL Initial Succeed ===========================\n");
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

int pass_size(int n , int f , int w , int h)
{
	return (6 + n*f*w*h/8);
}

void conn_server( unsigned char* payload, int size){
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

void pass_cloud(int model, uint8_t * bA, int layer_number, int m, int n, int w, int h){
	//clock_t package_start = clock();
        int f, data_size;		
        f = n / (w*h);
	data_size = pass_size(m, f, w, h);
        unsigned char * data = malloc(data_size * sizeof(unsigned char));

        data[0] = model;
	data[1] = layer_number;
	data[2] = m;
	data[3] = f;
	data[4] = w;
	data[5] = h;
        
        //  ====================== Encode ================================
        for (int i = 0; i < n/8 ; i++)
        {
                data[6+i] = 0;
                for (int j = i * 8; j < i * 8 + 8; j++)
                {
                        if (bA[j] == 1)
                        {
                                data[6+i] = data[6+i] << 1;
                                data[6+i] = data[6+i] + 1;
                        }
                        else
                        {
                                data[6+i] = data[6+i] << 1;
                        }
                }
        }
        
	//gettimeofday(&t0, NULL);
        conn_server(data , data_size);
        //gettimeofday(&t2, NULL);
        
        Server_CT = atof(mes[1]);
        Server_ET = atof(mes[2]);
        Final_ep = atoi(mes[0]);
	free(data);
}

//=========================================OpenCL Kernel=======================================

static void fconv_ocl(const float* A, const uint8_t* F, uint8_t* C,
        const float* Bias, const float* Gamma, const float* Beta,
        const float* Mean, const float* Std, const int m,
        const int num_f, const int w, const int h, const int d,
        const int kw, const int kh, const int sw, const int sh,
        const int pw, const int ph, const int pl_w,
        const int pl_h, const int pl_sw, const int pl_sh,
        const int pl_pw, const int pl_ph, uint8_t* bC, int max_m)
{
	cl_context context = context_init;
        cl_mem A_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, (w*h*d)* sizeof(A[0]), NULL, &ret);
        cl_mem F_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, (4 * num_f) * sizeof(F[0]), NULL, &ret);
        cl_mem Bias_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Bias[0]), NULL, &ret);
        cl_mem Gamma_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Gamma[0]), NULL, &ret);
        cl_mem Beta_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Beta[0]), NULL, &ret);
        cl_mem Mean_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Mean[0]), NULL, &ret);
        cl_mem Std_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Std[0]), NULL, &ret);

	cl_mem C_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, max_m * sizeof(uint8_t), NULL, &ret);
        cl_mem bC_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_f * w * h) * sizeof(bC[0]), NULL, &ret);

        ret = clEnqueueWriteBuffer(command_queue, A_mem, CL_TRUE, 0, (d*w*h)* sizeof(A[0]), A, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, F_mem, CL_TRUE, 0, (4 * num_f) * sizeof(F[0]), F, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Bias_mem, CL_TRUE, 0, num_f * sizeof(Bias[0]), Bias, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Gamma_mem, CL_TRUE, 0, num_f * sizeof(Gamma[0]), Gamma, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Beta_mem, CL_TRUE, 0, num_f * sizeof(Beta[0]), Beta, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Mean_mem, CL_TRUE, 0, num_f * sizeof(Mean[0]), Mean, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Std_mem, CL_TRUE, 0, num_f * sizeof(Std[0]), Std, 0, NULL, NULL);

	cl_kernel kernel = kernelf;

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&A_mem);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&F_mem);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&C_mem);
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&Bias_mem);
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&Gamma_mem);
        ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&Beta_mem);
        ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&Mean_mem);
        ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&Std_mem);

        ret = clSetKernelArg(kernel, 8, sizeof(cl_int), (void*)&m);
        ret = clSetKernelArg(kernel, 9, sizeof(cl_int), (void*)&num_f);
        ret = clSetKernelArg(kernel, 10, sizeof(cl_int), (void*)&w);
        ret = clSetKernelArg(kernel, 11, sizeof(cl_int), (void*)&h);
        ret = clSetKernelArg(kernel, 12, sizeof(cl_int), (void*)&d);
        ret = clSetKernelArg(kernel, 13, sizeof(cl_int), (void*)&kw);
        ret = clSetKernelArg(kernel, 14, sizeof(cl_int), (void*)&kh);
        ret = clSetKernelArg(kernel, 15, sizeof(cl_int), (void*)&sw);
        ret = clSetKernelArg(kernel, 16, sizeof(cl_int), (void*)&sh);
        ret = clSetKernelArg(kernel, 17, sizeof(cl_int), (void*)&pw);
        ret = clSetKernelArg(kernel, 18, sizeof(cl_int), (void*)&ph);
        ret = clSetKernelArg(kernel, 19, sizeof(cl_int), (void*)&pl_w);
        ret = clSetKernelArg(kernel, 20, sizeof(cl_int), (void*)&pl_h);
        ret = clSetKernelArg(kernel, 21, sizeof(cl_int), (void*)&pl_sw);
        ret = clSetKernelArg(kernel, 22, sizeof(cl_int), (void*)&pl_sh);
        ret = clSetKernelArg(kernel, 23, sizeof(cl_int), (void*)&pl_pw);
        ret = clSetKernelArg(kernel, 24, sizeof(cl_int), (void*)&pl_ph);

        ret = clSetKernelArg(kernel, 25, sizeof(cl_mem), (void *)&bC_mem);

	size_t global_item_size[2]; // Process the entire lists
        global_item_size[0] = m;
        global_item_size[1] = num_f;

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
        clFinish(command_queue);

	ret = clEnqueueReadBuffer(command_queue, C_mem, CL_TRUE, 0, max_m * sizeof(uint8_t), C, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, bC_mem, CL_TRUE, 0, num_f * w * h * sizeof(uint8_t), bC, 0, NULL, NULL);
}

void bconv_ocl(const uint8_t* A, const uint8_t* F, uint8_t* C,
        const float* Bias, const float* Gamma, const float* Beta,
        const float* Mean, const float* Std, const int m,
        const int num_f, const int w, const int h, const int d,
        const int kw, const int kh, const int sw, const int sh,
        const int pw, const int ph, const int pl_w,
        const int pl_h, const int pl_sw, const int pl_sh,
        const int pl_pw, const int pl_ph, uint8_t* bC, int max_m)
{
	cl_context context = context_init;
        cl_mem A_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, (w*h*num_f/8)* sizeof(A[0]), NULL, &ret);
        cl_mem F_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, (2 * num_f * num_f) * sizeof(F[0]), NULL, &ret);
        cl_mem Bias_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Bias[0]), NULL, &ret);
        cl_mem Gamma_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Gamma[0]), NULL, &ret);
        cl_mem Beta_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Beta[0]), NULL, &ret);
        cl_mem Mean_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Mean[0]), NULL, &ret);
        cl_mem Std_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, num_f * sizeof(Std[0]), NULL, &ret);

        cl_mem C_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, max_m * sizeof(uint8_t), NULL, &ret);
        cl_mem bC_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_f * w * h) * sizeof(bC[0]), NULL, &ret);

        ret = clEnqueueWriteBuffer(command_queue, A_mem, CL_TRUE, 0, (w*h*num_f / 8)* sizeof(A[0]), A, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, F_mem, CL_TRUE, 0, (2 * num_f * num_f) * sizeof(F[0]), F, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Bias_mem, CL_TRUE, 0, num_f * sizeof(Bias[0]), Bias, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Gamma_mem, CL_TRUE, 0, num_f * sizeof(Gamma[0]), Gamma, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Beta_mem, CL_TRUE, 0, num_f * sizeof(Beta[0]), Beta, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Mean_mem, CL_TRUE, 0, num_f * sizeof(Mean[0]), Mean, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, Std_mem, CL_TRUE, 0, num_f * sizeof(Std[0]), Std, 0, NULL, NULL);

	cl_kernel kernel = kernelb;

        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&A_mem);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&F_mem);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&C_mem);
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&Bias_mem);
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&Gamma_mem);
        ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&Beta_mem);
        ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&Mean_mem);
        ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&Std_mem);

        ret = clSetKernelArg(kernel, 8, sizeof(cl_int), (void*)&m);
        ret = clSetKernelArg(kernel, 9, sizeof(cl_int), (void*)&num_f);
        ret = clSetKernelArg(kernel, 10, sizeof(cl_int), (void*)&w);
        ret = clSetKernelArg(kernel, 11, sizeof(cl_int), (void*)&h);
        ret = clSetKernelArg(kernel, 12, sizeof(cl_int), (void*)&d);
        ret = clSetKernelArg(kernel, 13, sizeof(cl_int), (void*)&kw);
        ret = clSetKernelArg(kernel, 14, sizeof(cl_int), (void*)&kh);
        ret = clSetKernelArg(kernel, 15, sizeof(cl_int), (void*)&sw);
        ret = clSetKernelArg(kernel, 16, sizeof(cl_int), (void*)&sh);
        ret = clSetKernelArg(kernel, 17, sizeof(cl_int), (void*)&pw);
        ret = clSetKernelArg(kernel, 18, sizeof(cl_int), (void*)&ph);
        ret = clSetKernelArg(kernel, 19, sizeof(cl_int), (void*)&pl_w);
        ret = clSetKernelArg(kernel, 20, sizeof(cl_int), (void*)&pl_h);
        ret = clSetKernelArg(kernel, 21, sizeof(cl_int), (void*)&pl_sw);
        ret = clSetKernelArg(kernel, 22, sizeof(cl_int), (void*)&pl_sh);
        ret = clSetKernelArg(kernel, 23, sizeof(cl_int), (void*)&pl_pw);
        ret = clSetKernelArg(kernel, 24, sizeof(cl_int), (void*)&pl_ph);

        ret = clSetKernelArg(kernel, 25, sizeof(cl_mem), (void *)&bC_mem);

        size_t global_item_size[2]; // Process the entire lists
        global_item_size[0] = m;
        global_item_size[1] = num_f;

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
        clFinish(command_queue);

	ret = clEnqueueReadBuffer(command_queue, C_mem, CL_TRUE, 0, max_m * sizeof(uint8_t), C, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, bC_mem, CL_TRUE, 0, num_f * w * h * sizeof(uint8_t), bC, 0, NULL, NULL);
}




