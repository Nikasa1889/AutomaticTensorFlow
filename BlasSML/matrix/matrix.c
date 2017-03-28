#include <cblas.h>
#include <common.h>
#include "export.h"
#include "math.h"
//All functions excepts createCArr and toCArr do not use malloc in their bodies
//Function to create Arrray in C heap
Pointer createCArr(int l){
	double *a = malloc(l * sizeof(double));
	if (a == NULL) printf ("OUT OF MEMORY in createCArr function \n");
	return &a[0];
}

Pointer toCArr (double* x, int l){
	double *y = malloc( l * sizeof(double) );
	if (y == NULL) printf ("OUT OF MEMORY in toCArr function \n");
	memcpy(&y[0], &x[0], l * sizeof(double) );
	/*int i = 0;
	for (i = 0; i < l; i = i + 1){
		y[i] = x[i];
	} */
	return &y[0];
}

void printMat (double* x, int m, int n){
	int i, j = 0;
	for (i = 0; i < m; i = i + 1){
		for (j = 0; j < n; j = j + 1){
			printf("%f ", x[i*n + j]);
		}
		printf("\n");
	}
}
void printVec (double* x, int l){
	printMat (&x[0], 1, l);
}
double blas_ddot(int n, double* x, double* y){
	return cblas_ddot( n, &x[0], 1, &y[0], 1);
}
void blas_dcopy( int n, double* x, double* y){
	cblas_dcopy(n, &x[0], 1, &y[0], 1);
}
void blas_dscal( int n, double alpha, double* x){
	cblas_dscal( n, alpha, &x[0], 1);
}
void blas_axpy(int n, double alpha, double* x, double *y) {
    cblas_daxpy( n, alpha, &x[0], 1, &y[0], 1);
}
double blas_dnrm2(int n, double* x) {
	return cblas_dnrm2 (n, &x[0], 1);
}
/*void blas_dotmul(int n, double* x, double* y, double* z) {
	static const int k = 0; // Just the diagonal; 0 super-diagonal bands
    static const double alpha = 1.0;
    static const int lda = 1;
    static const double beta = 0.0;
	static const int inc_y = 1;
	static const int inc_z = 1;
		
	cblas_dsbmv(CblasRowMajor, CblasLower, n, k, alpha, &x[0], lda, &y[0], inc_y, beta, &z[0], inc_z );
} */
void blas_dotmul (int n, double* x, double* y, double* z) {
	int i = 0;
	for (i = 0; i < n; i = i + 1){
		z[i] = x[i] * y[i];
	}
}
void blas_reprows(int n, double* x, int m, double* Y){
	int i = 0;
	for (i = 0; i < m; i = i + 1){
		cblas_dcopy(n, &x[0], 1, &Y[i*n], 1);
	}
}

void blas_repcols(int m, double* x, int n, double* Y) {
	int j = 0;
	for (j = 0; j < n; j = j + 1) {
		cblas_dcopy(m, &x[0], 1, &Y[j], n);
	}
	/*
	int i = 0;
	int j = 0;
	for (i = 0; i < m; i = i +1){
		for ( j = 0; j < n; j = j + 1){
			Y[i*n+j] = x[i];
		}
	} */
}

void blas_dgemm(int isTransA, int isTransB, int m, int n, int k, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc){
	enum CBLAS_TRANSPOSE cblas_isTransA;
	enum CBLAS_TRANSPOSE cblas_isTransB;
	if (isTransA) cblas_isTransA = CblasTrans; else cblas_isTransA = CblasNoTrans; 
	if (isTransB) cblas_isTransB = CblasTrans; else cblas_isTransB = CblasNoTrans;
	cblas_dgemm( CblasRowMajor, cblas_isTransA, cblas_isTransB, m, n, k, alpha, &A[0], lda, &B[0], ldb, beta, &C[0], ldc);
}

void blas_dgemv(int isTransA, int m, int n, double alpha, double* A, double* x, double beta, double* y){
	enum CBLAS_TRANSPOSE cblas_isTransA;
	if (isTransA) cblas_isTransA = CblasTrans; else cblas_isTransA = CblasNoTrans;
	cblas_dgemv( CblasRowMajor, cblas_isTransA, m, n, alpha, &A[0], n, &x[0], 1, beta, &y[0], 1);
}
/*
void c_dapp(double* x, int l){
	int i = 0;
	for (i = 0; i < l; i = i + 1){
		x[i] = applyVecf( x[i] );
	}
} */
void c_dset(double* x, int l, double alpha){
	int i = 0;
	for (i = 0; i < l; i = i + 1){
		x[i] = alpha;
	}
	return &x[0];
}

void c_sumRows(double* A, int m, int n, double* x) {
	c_dset (&x[0], n, 0.0);
	int i = 0;
	for (i = 0; i < m; i = i + 1){
		blas_axpy(n, 1.0, &A[i*n], &x[0]);
	}
}
void c_sumCols(double* A, int m, int n, double* x) {
	c_dset (&x[0], m, 0.0);
	int j = 0;
	for (j = 0; j < n; j = j + 1){
		cblas_daxpy(m, 1.0, &A[j], n, &x[0], 1);
	}
}
double c_sum (double* x, int n) {
	int i = 0;
	double result = 0;
	for (i = 0; i < n; i = i + 1) {
		result = result + x[i];
	}
	return result;
}
void c_maxCols(double* A, int m, int n, double* x) {
	int i = 0;
	int j = 0;
	for ( i = 0; i < m; i = i + 1){
		x[i] = A[i*n];
		for (j = 1; j < n; j = j + 1) {
			if (x[i] < A[i*n + j]){
				x[i] = A[i*n + j];
			}
		}
	}
}
void c_maxColsIdx (double* A, int m, int n, double* x) {
	int i = 0;
	int j = 0;
	int bestIdx;
	for (i = 0; i < m; i = i + 1){
		bestIdx = i * n;
		for (j = 1; j < n; j = j + 1){
			if (A[bestIdx] < A[i * n + j]){
				bestIdx = i * n + j;
			}
		}
		x[i] = bestIdx - i * n;
	}
}

void c_mdscalRows(double* A, int m, int n, double* x) {
	int i = 0;
	for (i = 0; i < m; i = i + 1){
		cblas_dscal(n, x[i], &A[i*n], 1);
	}
	
}

int c_eqcount(int n, double* x, double* y){
	int i = 0;
	double eps = 0.0000000001;
	int count = 0;
	for ( i = 0; i < n; i = i + 1){
		if (fabs(x[i]-y[i]) < eps){
			count = count + 1;
		}
	}
	return count;
}

int c_eq(int n, double* x, double* y ){
	int i = 0;
	double eps = 0.0000000001;
	for ( i = 0; i < n; i = i + 1){
		if (fabs(x[i]-y[i]) > eps){
			return 0;
		}
	}
	return 1;
}


/* Extra functions for apply a specific function on matrix or vector */
double sigmoid (double x) {
	if (x > 13.0) return 1.0;
	else if (x < -13.0) return 0.0;
	else return 1.0/(1.0+exp(-x));
}
double dSigm (double x) { return x * (1.0 - x);}
double dTanh (double x) { return 1.0 - x*x;}
double hsquare (double x) { return 0.5 * x * x;}
double minusLn (double x) { return - log (x);}
double ce (double x) {return log(1.0 + exp(x));}
double invr (double x) {return 1.0/x;}
void c_dapp(double (*fptr)(double), double* x, int l){
	int i = 0;
	for (i = 0; i < l; i = i + 1){
		x[i] = fptr( x[i] );
	}
}

void dappSigm(double* x, int l) {c_dapp( &sigmoid, &x[0], l);}
void dappTanh(double* x, int l) {c_dapp( &tanh, &x[0], l);}
void dappDTanh(double* x, int l) {c_dapp( &dTanh, &x[0], l);}
void dappDSigm(double* x, int l) {c_dapp( &dSigm, &x[0], l);}
void dappExp(double* x, int l) {c_dapp(&exp, &x[0], l);}
void dappHSquare(double* x, int l) {c_dapp(&hsquare, &x[0], l);}
void dappMinusLn(double* x, int l) {c_dapp(&minusLn, &x[0], l);}
void dappCe (double* x, int l) {c_dapp(&ce, &x[0], l);}
void dappInvr (double* x, int l) {c_dapp(&invr, &x[0], l);}