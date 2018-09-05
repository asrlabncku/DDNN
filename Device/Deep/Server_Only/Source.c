#include <stdio.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>

#include "Tool_library.h"
struct timeval s_t0,s_t1;
struct timeval a_t0,a_t1;

#pragma warning( disable : 4996 )

int main(int argc , char* argv[])
{
  
  if(argc > 2){
      printf("Too many arguments supplied.\n");
      return 0;
  }

  float input[3*32*32];
  char *input_c[3*32*32];
  char filepath[50] = "../../dataset/cifar10_test_";
  char filename[50];
  uint8_t output[1];
  char num[8]; 
  char buff[200000];  

  //==========================Server Setup==============================
  gettimeofday(&s_t0, NULL);
  MQTT_initialization();
  gettimeofday(&s_t1, NULL);

  float Service_f_0 = (float)s_t0.tv_usec;
  float Service_f_1 = (float)s_t1.tv_usec;
  for(;Service_f_0 > 1;Service_f_0 /= 10.0f){}
  for(;Service_f_1 > 1;Service_f_1 /= 10.0f){}
  float Service_T = (float)(s_t1.tv_sec - s_t0.tv_sec) + (Service_f_1 - Service_f_0);
  
  printf("Service_Setup_T : %f \n",Service_T);

  //===========================Inference=================================
  for(int i = 0 ; i < atoi(argv[1]) ; i++)
  {
        filename[0] = '\0';
        strcat(filename, filepath);
        sprintf(num, "%d", i);
        strcat(filename, num);
        strcat(filename, ".txt");
        //printf("%s\n" , filename);

        FILE *fp = fopen(filename, "r");
    	if (!fp) {
          printf("Filename : %s\n",filename);
          fprintf(stderr, "Failed to load and error code :%d\n",errno);
          //system("pause");
          return(0);
  	}
        fscanf(fp, "%s", buff);
  	fclose(fp);
        
        split(input_c, buff, ",");

  	for (int i = 0; i < 3 * 32 * 32; i++)
          input[i] = atof(input_c[i]);
	
        gettimeofday(&a_t0, NULL);
        pass_cloud_raw(6, input , 0 , 1 , 3 ,32 , 32);
        gettimeofday(&a_t1, NULL);

	float All_f_0 = (float)a_t0.tv_usec;
        float All_f_1 = (float)a_t1.tv_usec;
        for(;All_f_0 > 1;All_f_0 /= 10.0f){}
        for(;All_f_1 > 1;All_f_1 /= 10.0f){}
        float All_T = (float)(a_t1.tv_sec - a_t0.tv_sec) + (All_f_1 - All_f_0);
        
        float Pass_f_0 = (float)c_t0.tv_usec;
        float Pass_f_1 = (float)c_t1.tv_usec;
        for(;Pass_f_0 > 1;Pass_f_0 /= 10.0f){}
        for(;Pass_f_1 > 1;Pass_f_1 /= 10.0f){}
        float passing = (float)(c_t1.tv_sec - c_t0.tv_sec) + (Pass_f_1 - Pass_f_0);
        Pass_T = passing - Server_CT;

        All_T = All_T - 0.02;
        Server_CT = Server_CT - 0.02;

	printf("Pass_T : %f \t Server_CT : %f \t Server_ET : %f \t Fp : %d \t All_T : %f\n",Pass_T,Server_CT,Server_ET,Final_ep,All_T);
 
	FILE *frs = fopen("result_raw.txt", "a+");
  	fprintf(frs,"Pass_T : %f \t Server_CT : %f \t Server_ET : %f \t Fp : %d \t All_T : %f\n",Pass_T,Server_CT,Server_ET,Final_ep,All_T);
  }

  MQTTClient_disconnect(client, 10000);
  MQTTClient_destroy(&client);

  return 0;
}

