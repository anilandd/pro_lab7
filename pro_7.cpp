
#include<windows.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<conio.h>
#include<math.h>
#include<time.h>
#include <synchapi.h>
#include <iostream>
#define NRA 12 /* number of rows in matrix A */
#define NCA 12 /* number of columns in matrix A */
#define NCB 12 /* number of columns in matrix B */
#define MASTER 0 /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */
using namespace std;
double case2();
double case1();
int ProcNum;
int ProcRank;
int Size;
double* A;  double* B; double* C;

void printMatrix(double* pMatrix, int Size) {
	for (int i = 0; i < Size; i++) {
		cout << endl;
		for (int j = 0; j < Size; j++)
			cout << pMatrix[i * Size + j] << " ";
	}
}

void getMatrix(double* pMatrix, int Size) {
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++)  pMatrix[i * Size + j] = 10;
	}
}

void initProcess(double*& A, double*& B, double*& C, int& Size) {
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	if (ProcRank == 0) {
		do {
			Size = NRA;
			if (Size < ProcNum) printf("Matrix size is less than the number of processes! \n");
			if (Size % ProcNum != 0) printf("Matrix size should be dividable by the number of processes! \n");
		} while ((Size < ProcNum) || (Size % ProcNum != 0));
	}

	MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (ProcRank == 0) {
		A = new double[Size * Size];
		B = new double[Size * Size];
		C = new double[Size * Size];
		getMatrix(A, Size);
		getMatrix(B, Size);
	}
}

void flip(double*& B, int dim) {
	double temp = 0.0;
	for (int i = 0; i < dim; i++) {
		for (int j = i + 1; j < dim; j++) {
			temp = B[i * dim + j]; 
			B[i * dim + j] = B[j * dim + i]; 
			B[j * dim + i] = temp;
		}
	}
}

void matrixMultiplicationMPI(double*& A, double*& B, double*& C, int& Size) {
	int dim = Size;
	int i, j, k, p, ind;
	double temp;
	MPI_Status Status;
	int ProcPartSize = dim / ProcNum;
	int ProcPartElem = ProcPartSize * dim;
	double* bufA = new double[dim * ProcPartSize];
	double* bufB = new double[dim * ProcPartSize];
	double* bufC = new double[dim * ProcPartSize];
	int ProcPart = dim / ProcNum, part = ProcPart * dim;
	if (ProcRank == 0) {
		flip(B, Size);
	}

	MPI_Scatter(A, part, MPI_DOUBLE, bufA, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, part, MPI_DOUBLE, bufB, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	temp = 0.0;
	for (i = 0; i < ProcPartSize; i++) {
		for (j = 0; j < ProcPartSize; j++) {
			for (k = 0; k < dim; k++) temp += bufA[i * dim + k] * bufB[j * dim + k];
			bufC[i * dim + j + ProcPartSize * ProcRank] = temp; 
			temp = 0.0;
		}
	}

	int NextProc; int PrevProc;
	for (p = 1; p < ProcNum; p++) {
		NextProc = ProcRank + 1;
		if (ProcRank == ProcNum - 1) NextProc = 0;
		PrevProc = ProcRank - 1;
		if (ProcRank == 0) PrevProc = ProcNum - 1;
		MPI_Sendrecv_replace(bufB, part, MPI_DOUBLE, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
		temp = 0.0;
		for (i = 0; i < ProcPartSize; i++) {
			for (j = 0; j < ProcPartSize; j++) {
				for (k = 0; k < dim; k++) {
					temp += bufA[i * dim + k] * bufB[j * dim + k];
				}
				if (ProcRank - p >= 0)
					ind = ProcRank - p;
				else ind = (ProcNum - p + ProcRank);
				bufC[i * dim + j + ind * ProcPartSize] = temp;
				temp = 0.0;
			}
		}
	}
	MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[]bufA;
	delete[]bufB;
	delete[]bufC;
}

double case2() {
	int numtasks,
		taskid,
		numworkers,
		source,
		dest,
		rows, /* rows of matrix A sent to each worker */
		averow, extra, offset,
		i, j, k, rc;
	double a[NRA][NCA], /* matrix A to be multiplied */
		b[NCA][NCB], /* matrix B to be multiplied */
		c[NRA][NCB]; /* result matrix C */
	MPI_Status status;
	//MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	if (numtasks < 2) {
		printf("Need at least two MPI tasks. Quitting...\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(1);
	}
	numworkers = numtasks - 1;

	double start_time = 0;
	double end_time = 0;

	if (taskid == MASTER) {
		printf("mpi_mm has started with %d tasks.\n", numtasks);
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCA; j++)
				a[i][j] = 10;
		for (i = 0; i < NCA; i++)
			for (j = 0; j < NCB; j++)
				b[i][j] = 10;

		averow = NRA / numworkers;
		extra = NRA % numworkers;
		offset = 0;

		start_time = MPI_Wtime();

		for (dest = 1; dest <= numworkers; dest++) {
			rows = (dest <= extra) ? averow + 1 : averow;
			printf("Sending %d rows to task %d offset= % d\n", rows, dest, offset);
			MPI_Request send_req;

			MPI_Isend(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req);
			Sleep(0);
			MPI_Wait(&send_req, MPI_STATUS_IGNORE);

			MPI_Isend(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req);
			Sleep(0);
			MPI_Wait(&send_req, MPI_STATUS_IGNORE);

			MPI_Isend(&a[offset][0], rows * NCA, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req);
			Sleep(0);
			MPI_Wait(&send_req, MPI_STATUS_IGNORE);

			MPI_Isend(&b, NCA * NCB, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD, &send_req);
			Sleep(0);
			MPI_Wait(&send_req, MPI_STATUS_IGNORE);

			offset = offset + rows;
		}
		/* Receive results from worker tasks */
		for (source = 1; source <= numworkers; source++) {
			MPI_Request recv_req;

			MPI_Irecv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &recv_req);
			Sleep(0);
			MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

			MPI_Irecv(&rows, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &recv_req);
			Sleep(0);
			MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

			MPI_Irecv(&c[offset][0], rows * NCB, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &recv_req);
			Sleep(0);
			MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

			printf("Received results from task %d\n", taskid);
		}

		end_time = MPI_Wtime();

		/* Print results */
		printf("****\n");
		printf("Result Matrix NonBlocked:\n");
		for (i = 0; i < NRA; i++) {
			printf("\n");
			for (j = 0; j < NCB; j++)
				printf("%6.2f ", c[i][j]);
		}
		printf("\n********\n");
		printf("Done.\n");
	}
	/******** worker task *****************/
	else { /* if (taskid > MASTER) */
		MPI_Request recv_req;

		MPI_Irecv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_req);
		Sleep(0);
		MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

		MPI_Irecv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_req);
		Sleep(0);
		MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

		MPI_Irecv(&a, rows * NCA, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_req);
		Sleep(0);
		MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

		MPI_Irecv(&b, NCA * NCB, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_req);
		Sleep(0);
		MPI_Wait(&recv_req, MPI_STATUS_IGNORE);

		for (k = 0; k < NCB; k++)
			for (i = 0; i < rows; i++) {
				c[i][k] = 0.0;
				for (j = 0; j < NCA; j++)
					c[i][k] = c[i][k] + a[i][j] * b[j][k];
			}

		MPI_Request send_req;

		MPI_Isend(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD, &send_req);
		Sleep(0);
		MPI_Wait(&send_req, MPI_STATUS_IGNORE);

		MPI_Isend(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD, &send_req);
		Sleep(0);
		MPI_Wait(&send_req, MPI_STATUS_IGNORE);

		MPI_Isend(&c, rows * NCB, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD, &send_req);
		Sleep(0);
		MPI_Wait(&send_req, MPI_STATUS_IGNORE);
	}

	return end_time - start_time;
	//MPI_Finalize();
}

double case1() {
	int numtasks,
		taskid,
		numworkers,
		source,
		dest,
		rows, /* rows of matrix A sent to each worker */
		averow, extra, offset,
		i, j, k, rc;
	double a[NRA][NCA], /* matrix A to be multiplied */
		b[NCA][NCB], /* matrix B to be multiplied */
		c[NRA][NCB]; /* result matrix C */
	MPI_Status status;
	//MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	if (numtasks < 2) {
		printf("Need at least two MPI tasks. Quitting...\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(1);
	}
	numworkers = numtasks - 1;

	double start_time = 0;
	double end_time = 0;

	if (taskid == MASTER) {
		printf("mpi_mm has started with %d tasks.\n", numtasks);
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCA; j++)
				a[i][j] = 10;
		for (i = 0; i < NCA; i++)
			for (j = 0; j < NCB; j++)
				b[i][j] = 10;

		averow = NRA / numworkers;
		extra = NRA % numworkers;
		offset = 0;

		start_time = MPI_Wtime();

		for (dest = 1; dest <= numworkers; dest++) {
			rows = (dest <= extra) ? averow + 1 : averow;
			printf("Sending %d rows to task %d offset= % d\n", rows, dest, offset
			);
			MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER,
				MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER,
				MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows * NCA, MPI_DOUBLE, dest,
				FROM_MASTER, MPI_COMM_WORLD);
			MPI_Send(&b, NCA * NCB, MPI_DOUBLE, dest, FROM_MASTER,
				MPI_COMM_WORLD);
			offset = offset + rows;
		}
		/* Receive results from worker tasks */
		for (source = 1; source <= numworkers; source++) {
			MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER,
				MPI_COMM_WORLD,
				&status);
			MPI_Recv(&rows, 1, MPI_INT, source, FROM_WORKER,
				MPI_COMM_WORLD,
				&status);
			MPI_Recv(&c[offset][0], rows * NCB, MPI_DOUBLE,
				source,
				FROM_WORKER, MPI_COMM_WORLD,
				&status);
			printf("Received results from task %d\n", taskid);
		}

		end_time = MPI_Wtime();

		/* Print results */
		printf("****\n");
		printf("Result Matrix Blocked:\n");
		for (i = 0; i < NRA; i++) {
			printf("\n");
			for (j = 0; j < NCB; j++)
				printf("%6.2f ", c[i][j]);
		}
		printf("\n********\n");
		printf("Done.\n");
	}
	/******** worker task *****************/
	else { /* if (taskid > MASTER) */
		MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER,
			MPI_COMM_WORLD,
			&status);
		MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER,
			MPI_COMM_WORLD, &status);
		MPI_Recv(&a, rows * NCA, MPI_DOUBLE, MASTER, FROM_MASTER,
			MPI_COMM_WORLD,
			&status);
		MPI_Recv(&b, NCA * NCB, MPI_DOUBLE, MASTER, FROM_MASTER,
			MPI_COMM_WORLD,
			&status);
		for (k = 0; k < NCB; k++)
			for (i = 0; i < rows; i++) {
				c[i][k] = 0.0;
				for (j = 0; j < NCA; j++)
					c[i][k] = c[i][k] + a[i][j] * b[j][k];
			}
		MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER,
			MPI_COMM_WORLD);
		MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER,
			MPI_COMM_WORLD);
		MPI_Send(&c, rows * NCB, MPI_DOUBLE, MASTER,
			FROM_WORKER, MPI_COMM_WORLD);
	}

	return end_time - start_time;
	//MPI_Finalize();
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	double blockedTime = case1();
	double nonBlockedTime = case2();
	double beg, end, serial, parallel = 0;
	initProcess(A, B, C, Size);
	beg = MPI_Wtime();
	matrixMultiplicationMPI(A, B, C, Size);
	end = MPI_Wtime(); 
	parallel = end - beg;
	if (ProcRank == 0) {
		cout << "\nMatrix C - Parallel calculation\n";
		printMatrix(C, Size);
		cout << endl << "Time of execution - Parallel calculation : " << endl;
		cout << "parallel: " << parallel << endl;
		cout << "blockedTime: " << blockedTime << endl;
		cout << "nonBlockedTime: " << nonBlockedTime << endl;
		cout << "efficiency 1: " << blockedTime / parallel << endl;
		cout << "efficiency 2: " << nonBlockedTime / parallel << endl;
	}
	MPI_Finalize();
	delete[] A; delete[] B; delete[] C;
	return 0;
}