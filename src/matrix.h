#ifndef _MATRIX_H_
#define _MATRIX_H_

#pragma once

#include <stdbool.h>

typedef struct st_matrix {
    double *mtx;
    size_t nrow;
    size_t ncol;
} st_matrix;

void init_st_matrix(st_matrix *mtx, size_t nrow, size_t ncol);

st_matrix* build_matrix(double *vec, size_t nrow, size_t ncol);

void free_st_matrix(st_matrix *mtx);

void mtxcpy(st_matrix *dest, st_matrix *src);

double get(st_matrix *mtx, size_t row, size_t col);

double* getp(st_matrix *mtx, size_t row, size_t col);

double sget(st_matrix *mtx, size_t row, size_t col);

void set(st_matrix *mtx, size_t row, size_t col, double val);

st_matrix* setall(st_matrix *mtx, double val);

st_matrix* identity(size_t n);

st_matrix* mtxadd(st_matrix *a, st_matrix *b);

st_matrix* mtxadd_(st_matrix *dest, st_matrix *a, st_matrix *b);

st_matrix* mtxsub(st_matrix *a, st_matrix *b);

st_matrix* mtxsub_(st_matrix *dest, st_matrix *a, st_matrix *b);

st_matrix* mtxmult(st_matrix *a, st_matrix *b);

st_matrix* mtxmult_(st_matrix *dest, st_matrix *a, st_matrix *b);

st_matrix* mtxsmult(st_matrix *mtx, double val);

st_matrix* mtxsmult_(st_matrix *dest, st_matrix *src, double val);

st_matrix* mtxid(st_matrix *mtx);

st_matrix* transpose(st_matrix *mtx);

st_matrix* transpose_(st_matrix *dest, st_matrix *src);

bool mtxeq(st_matrix *mtx1, st_matrix *mtx2);

void print_st_matrix(st_matrix *mtx, size_t prec, bool lines);

#endif /* _MATRIX_H_ */
