typedef uchar uint8_t;

#define CEIL_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define MAX_FILTER_BYTES 12
#define MIN(a,b) ((a) < (b) ? a : b)
#define MAX(a,b) ((a) > (b) ? a : b)

#define BIT_SIZE 8
uint8_t constant bits[BIT_SIZE] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};

void fconv(__global float* A, __global uint8_t* F,__global uint8_t* C,const int c_start_idx, const  float Bias, const  float Gamma,
	const  float Beta, const  float Mean, const  float Std,const int w, const int h, const int d, const int kw,
	const int kh, const int sw, const int sh, const int pw,const int ph, const int pl_w, const int pl_h, const int pl_sw,
	const int pl_sh, const int pl_pw, const int pl_ph , __global uint8_t* bC);

float batch_norm(float f, const float Gamma, const float Beta, const float Mean, const float Std);

int convpool_size(const int x, const int kx, const int sx, const int px, const int pl_x, const int pl_sx, const int pl_px);

int conv_idx(const int pl_i, const int x, const int kx, const int sx, const int px);

float fdot_3d(__global float* A, __global uint8_t* B, const int x, const int y,const int w, const int h, const int d, 
	const int kw,const int kh);

int idx_2d(const int i, const int j, const int rows);

int nthbitset_arr(__global uint8_t* const arr, const int n);

uint8_t rotr1 (const uint8_t x);


__kernel void conv_pool_f(__global float *A , __global uint8_t *F , __global uint8_t *C , __global float *Bias
,__global float *Gamma , __global float *Beta,__global float *Mean , __global float *Std,int m 
,int num_f , int w , int h , int d , int kw , int kh , int sw , int sh , int pw , int ph , int pl_w , int pl_h
 , int pl_sw , int pl_sh , int pl_pw , int pl_ph , __global uint8_t* bC) {

	int i = get_global_id(0);//0
	int j = get_global_id(1);//0 or 1

    int res_size, c_idx, a_idx, f_idx;
    
/*
	for (int t = 0; t < 50; t++)
		 printf("A[%d] = %f \n", 20 * t, A[20 * t]);
	for (int i = 0; i < 4 * num_f; i++)
		printf("F[%d] = %u \n", i, F[i]);
	for (int i = 0; i < num_f; i++)
		printf("Bias[%d] = %f \n ", i, Bias[i]);
	for (int i = 0; i < num_f; i++)
		printf("Gamma[%d] = %f \n ", i, Gamma[i]);
	for (int i = 0; i < num_f; i++)
		printf("Beta[%d] = %f \n ", i, Beta[i]);
	for (int i = 0; i < num_f; i++)
		printf("Mean[%d] = %f \n ", i, Mean[i]);
	for (int i = 0; i < num_f; i++)
		printf("Std[%d] = %f \n ", i, Std[i]);
*/
    
	res_size = w * h;
	c_idx = res_size*j;
	a_idx = i*w*h*d;
	f_idx = j*CEIL_POS(kw*kh*d / 8.0);

	fconv(A + a_idx, F + f_idx, C, c_idx, Bias[j], Gamma[j],
		Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
		pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph, bC);

	/*for (int p = 0; p < num_f; p++)
	{
		printf("X[%d] (binary) input : \n", p);
		for (int q = 0; q < w * h; q++)
		{
			printf(",%d", bC[q + (w*h*p)]);
			if ((q % w) == (w - 1))
				printf("\n");
		}
		printf("\n");
	}*/

}   


void fconv(__global float* A, __global uint8_t* F, __global uint8_t* C, const int c_start_idx, const float Bias, const float Gamma,
	const float Beta, const float Mean, const float Std, const int w, const int h, const int d, const int kw,
	const int kh, const int sw, const int sh, const int pw, const int ph, const int pl_w, const int pl_h, const int pl_sw,
	const int pl_sh, const int pl_pw, const int pl_ph , __global uint8_t* bC){

	uint8_t c_mask, res_sign;
	int pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, c_shift, c_idx;
	float res, max_res;

	c_shift = 7 - (c_start_idx % 8);
	c_mask = 1 << c_shift;
	c_idx = c_start_idx / 8;
	pl_i_max = (w - kw + 2 * pw) / sw + (2 * pl_pw) + 1;
	pl_j_max = (h - kh + 2 * ph) / sh + (2 * pl_ph) + 1;

	int num = 0;
	for (pl_i = -pl_pw; pl_i + pl_w + pl_pw - 1 < pl_i_max; pl_i += pl_sw) {
		//printf("Hi !!!!!\n");
		for (pl_j = -pl_ph; pl_j + pl_h + pl_pw - 1 < pl_j_max; pl_j += pl_sh) {
			max_res = res = -FLT_MAX;
			for (i_in = pl_i; i_in < pl_i + pl_w; ++i_in) {
				i = conv_idx(i_in, w, kw, sw, pw);
				for (j_in = pl_j; j_in < pl_j + pl_h; ++j_in) {
					j = conv_idx(j_in, h, kh, sh, ph);
					if (i >= -pw && j >= -ph) {
						res = fdot_3d(A, F, i, j, w, h, d, kw, kh);
						max_res = MAX(res, max_res);
					}
				}
			}
			max_res += Bias;
			max_res = batch_norm(max_res, Gamma, Beta, Mean, Std);

			res_sign = max_res >= 0 ? 1 : 0;

			/* store result */
			bC[c_start_idx + num] = res_sign;
			C[c_idx] |= res_sign << c_shift;
			//printf("C[%d] = %u \n", c_idx, C[c_idx]);
			/* update c_idx */
			c_mask = rotr1(c_mask);
			c_idx += (c_mask & 0x80) >> 7;
			c_shift--;
			c_shift = c_shift < 0 ? 7 : c_shift;
			num++;
		}
	}
}



float batch_norm(float f, const float Gamma, const float Beta, const float Mean, const float Std){
	f -= Mean;
	f /= Std;
	f *= Gamma;
	f += Beta;
	return f;
}

uint8_t rotr1(const uint8_t x){
	return (x >> 1) | (x << 7);
}

int conv_idx(const int pl_i, const int x, const int kx, const int sx, const int px){
	int conv_sz = (x - kx + 2 * px) / sx;
	return (pl_i < 0 || pl_i > conv_sz) ? -INT_MAX : pl_i * sx - px;
}

float fdot_3d(__global float* A, __global uint8_t* B, const int x, const int y, const int w, const int h, const int d, const int kw,
	const int kh){
	uint8_t  bitset;
	int i, j, k, b_idx, A_bytes;
	float a, res;
	__global float *A_slice;
	A_bytes = w*h;
	res = 0;
	b_idx = 0;

	for (i = 0; i < d; ++i) {  // d = 1
		A_slice = A + A_bytes*i;
		for (j = x; j < x + kw; ++j) { // j = x ~ x+3
			for (k = y; k < y + kh; ++k) { // k = y ~ y+3
				/* handles padding */
				if (j < 0 || j > h - 1 || k < 0 || k > w - 1) {
					a = 0.0;
				}
				else {
					a = A_slice[idx_2d(j, k, w)];
				}
				bitset = nthbitset_arr(B, b_idx);
				res += bitset ? a : -a;
				b_idx++;
			}
		}
	}
	return res;
}

int idx_2d(const int i, const int j, const int rows){
	return i * rows + j;
}

int nthbitset_arr(__global uint8_t* const arr, const int n){
	return arr[n / 8] & bits[n % 8] ? 1 : 0;
}
