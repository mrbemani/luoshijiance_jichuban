#include <stdio.h>
#include <math.h>

#define MATRIX_SIZE 2

typedef struct {
    double matrix[MATRIX_SIZE][MATRIX_SIZE];
} Matrix;

typedef struct {
    double vector[MATRIX_SIZE];
} Vector;

void matrixMultiplication(Matrix *a, Matrix *b, Matrix *result)
{
    int i, j, k;
    for(i = 0; i < MATRIX_SIZE; i++) {
        for(j = 0; j < MATRIX_SIZE; j++) {
            result->matrix[i][j] = 0;
            for(k = 0; k < MATRIX_SIZE; k++) {
                result->matrix[i][j] += a->matrix[i][k] * b->matrix[k][j];
            }
        }
    }
}

void vectorMultiplication(Matrix *a, Vector *b, Vector *result)
{
    int i, j;
    for(i = 0; i < MATRIX_SIZE; i++) {
        result->vector[i] = 0;
        for(j = 0; j < MATRIX_SIZE; j++) {
            result->vector[i] += a->matrix[i][j] * b->vector[j];
        }
    }
}

void matrixTranspose(Matrix* m, Matrix* result)
{
    int i, j;
    for(i = 0; i < MATRIX_SIZE; i++) {
        for(j = 0; j < MATRIX_SIZE; j++) {
            result->matrix[i][j] = m->matrix[j][i];
        }
    }
}

void inverse(Matrix* m, Matrix* result)
{
    double determinant = m->matrix[0][0] * m->matrix[1][1] - m->matrix[0][1] * m->matrix[1][0];

    result->matrix[0][0] = m->matrix[1][1];
    result->matrix[0][1] = -m->matrix[0][1];
    result->matrix[1][0] = -m->matrix[1][0];
    result->matrix[1][1] = m->matrix[0][0];

    int i, j;
    for(i = 0; i < MATRIX_SIZE; i++) {
        for(j = 0; j < MATRIX_SIZE; j++) {
            result->matrix[i][j] /= determinant;
        }
    }
}

void printMatrix(Matrix* m)
{
    int i, j;
    for(i = 0; i < MATRIX_SIZE; i++) {
        for(j = 0; j < MATRIX_SIZE; j++) {
            printf("%f ", m->matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printVector(Vector* v)
{
    int i;
    for(i = 0; i < MATRIX_SIZE; i++) {
        printf("%f ", v->vector[i]);
    }
    printf("\n");
}

void predict(Matrix *F, Vector *u, Matrix *P, Vector *lastResult)
{
    Matrix tempMatrix;
    Vector tempVector;

    matrixMultiplication(F, u, &tempVector);
    vectorMultiplication(&tempVector, lastResult, &tempVector);

    matrixMultiplication(F, P, &tempMatrix);
    matrixTranspose(F, &tempMatrix);
    matrixMultiplication(&tempMatrix, F, P);

    lastResult->vector[0] = u->vector[0];
    lastResult->vector[1] = u->vector[1];
}

void correct(Matrix *A, Matrix *P, Matrix *R, Matrix *b, Vector *u, Vector *lastResult, int flag)
{
    Matrix tempMatrix;
    Vector tempVector;

    if(!flag) {
        u->vector[0] = lastResult->vector[0];
        u->vector[1] = lastResult->vector[1];
        matrixMultiplication(A, P, &tempMatrix);
        matrixTranspose(A, &tempMatrix);
        matrixMultiplication(&tempMatrix, A, &tempMatrix);
        matrixMultiplication(&tempMatrix, R, &tempMatrix);
        inverse(&tempMatrix, &tempMatrix);
        matrixMultiplication(P, &tempMatrix, P);
    }
    else {
        matrixMultiplication(A, P, &tempMatrix);
        matrixTranspose(A, &tempMatrix);
        matrixMultiplication(&tempMatrix, A, &tempMatrix);
        matrixMultiplication(&tempMatrix, R, &tempMatrix);
        inverse(&tempMatrix, &tempMatrix);

        vectorMultiplication(A, u, &tempVector);
        tempVector.vector[0] = b->matrix[0][0] - tempVector.vector[0];
        tempVector.vector[1] = b->matrix[1][0] - tempVector.vector[1];
        matrixMultiplication(&tempMatrix, &tempVector, &tempVector);
        vectorMultiplication(P, &tempVector, &tempVector);
        lastResult->vector[0] += tempVector.vector[0];
        lastResult->vector[1] += tempVector.vector[1];
    }
}

int main()
{
    Matrix F = {1, 0, 0, 1};
    Vector u = {0, 0};
    Matrix P = {1, 0, 0, 1};
    Vector lastResult = {0, 0};

    Matrix A = {1, 0, 0, 1};
    Matrix R = {1, 0, 0, 1};
    Matrix b = {1, 1};
    int flag = 0;

    predict(&F, &u, &P, &lastResult);
    correct(&A, &P, &R, &b, &u, &lastResult, flag);

    printVector(&lastResult);

    return 0;
}


