
#include <iostream>
#include <complex>
#include <assert.h>
#include "omp.h"
#include <cmath>
#include "time.h"
#include "sys/time.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;


typedef complex<double> complexd;



complexd* generate_condition(int n, long long unsigned count){ 
	double module = 0;
	complexd *V = new complexd[count];
	#pragma omp parallel shared(V) reduction(+: module)
	{
		#pragma omp for schedule(static)
		for (long long unsigned  i = 0; i < count; i++){
			V[i].real(rand());
			V[i].imag(rand());
			module += abs(V[i] * V[i]);
		}
	}
	module = sqrt(module);
	#pragma omp parallel shared(count)
	{
		#pragma omp for schedule(static)
    	for (long long unsigned j = 0; j < count; j++) {
        	V[j] /= module;
    	}
    }


	return V;
}





void OneQubitEvolution(complexd *in,complexd *out,complexd U[2][2],int n, int q){
    int shift = n - q;
    int pow = 1 << (shift);
    int N = 1 << n;
#pragma omp parallel 
{
	#pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
        int i0 = i & ~pow;
        int i1 = i | pow;
        int iq = (i & pow) >> shift;
        out[i] = U[iq][0] * in[i0] + U[iq][1] * in[i1];
        }
    }
}





int main(int argc , char** argv) {
    int n = atoi(argv[1]); 
    int k = atoi(argv[2]); 
    struct timeval start,stop;



    long long unsigned count = 1;
	#pragma omp parallel reduction(*: count)
	{
		#pragma omp for schedule(static)
		for (int i = 0; i < n; i++){
			count = count * 2;
		}
	}



    complexd *V = generate_condition(n, count);
    
    complexd *W = new complexd[count];
    complexd U[2][2];
    U[0][0] = 1/sqrt(2);
    U[0][1] = 1/sqrt(2);
    U[1][0] = 1/sqrt(2);
    U[1][1] =  - 1/sqrt(2);
    gettimeofday(&start,NULL);
    OneQubitEvolution(V,W,U,n,k);
    gettimeofday(&stop,NULL);
   // for (long long unsigned i = 0; i < count; i++){
 //   	cout << W[i] << endl;
    //}
    cout << sizeof(V[1]) << endl;
    printf("%lf\n",(float)((stop.tv_sec - start.tv_sec)*1000000 + stop.tv_usec - start.tv_usec)/1000000);
    delete[] V;
    delete[] W;
}