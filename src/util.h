#ifndef _UTIL_H_
#define _UTIL_H_

#pragma once

#include <stdbool.h>

#include "matrix.h"

#define FPOINT_OFFSET 1e-9

bool load_data(char *fname, st_matrix *mtx);

// weightless minkowski
double minkowski(double *a, double *b, size_t size, double p);

double sqdeuclid_dist(double *a, double *b, size_t size);

double euclid_dist(double *a, double *b, size_t size);

void print_mtx_d_(double **matrix, size_t nrow, size_t ncol);

void print_mtx_d(double *mtx, size_t nrow, size_t ncol);

void fprint_mtx_d(FILE *file, double **matrix, size_t nrow,
					size_t ncol);

void print_mtx_size_t(size_t **matrix, size_t nrow, size_t ncol);

void print_mtx_int(int *mtx, size_t nrow, size_t ncol);

bool deq(double a, double b);

bool dgt(double a, double b);

bool dlt(double a, double b);

int cmpint(const void *a, const void *b);

int max(int *vec, size_t size);

void print_header(char *str, size_t size);

void mtxcpy_d(double **destination, double **source, size_t nrow,
        size_t ncol);

void mtxcpy_size_t(size_t **destination, size_t **source, size_t nrow,
        size_t ncol);

#endif /* _UTIL_H_ */
