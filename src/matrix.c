#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "util.h"
#include "matrix.h"

void init_st_matrix(st_matrix *mtx, size_t nrow, size_t ncol) {
    mtx->mtx = malloc(sizeof(double) * nrow * ncol);
    mtx->nrow = nrow;
    mtx->ncol = ncol;
}

void free_st_matrix(st_matrix *mtx) {
    free(mtx->mtx);
}

st_matrix* build_matrix(double *vec, size_t nrow, size_t ncol) {
    st_matrix *ret = malloc(sizeof(st_matrix));
    init_st_matrix(ret, nrow, ncol);
    memcpy(ret->mtx, vec, sizeof(double) * nrow * ncol);
    return ret;
}

void mtxcpy(st_matrix *dest, st_matrix *src) {
    memcpy(dest->mtx, src->mtx, sizeof(double) * src->nrow *
            src->ncol);
    dest->nrow = src->nrow;
    dest->ncol = src->ncol;
}

double get(st_matrix *mtx, size_t row, size_t col) {
    return mtx->mtx[row * mtx->ncol + col];
}

double* getp(st_matrix *mtx, size_t row, size_t col) {
    return &(mtx->mtx[row * mtx->ncol + col]);
}

double sget(st_matrix *mtx, size_t row, size_t col) {
    if(row >= mtx->nrow || col >= mtx->ncol) {
        return NAN;
    }
    return mtx->mtx[row * mtx->ncol + col];
}

void set(st_matrix *mtx, size_t row, size_t col, double val) {
    mtx->mtx[row * mtx->ncol + col] = val;
}

st_matrix* setall(st_matrix *mtx, double val) {
    size_t i;
    size_t j;
    for(i = 0; i < mtx->nrow; ++i) {
        for(j = 0; j < mtx->ncol; ++j) {
            set(mtx, i, j, val);
        }
    }
    return mtx;
}

st_matrix* identity(size_t n) {
    st_matrix *id = malloc(sizeof(st_matrix));
    id->mtx = calloc(sizeof(double), n * n);
    id->nrow = n;
    id->ncol = n;
    size_t i;
    for(i = 0; i < n; ++i) {
        set(id, i, i, 1.0);
    }
    return id;
}

st_matrix* mtxadd(st_matrix *a, st_matrix *b) {
    st_matrix *ret = malloc(sizeof(st_matrix));
    init_st_matrix(ret, a->nrow, a->ncol);
    if(!mtxadd_(ret, a, b)) {
        free_st_matrix(ret);
        free(ret);
        return NULL;
    }
    return ret;
}

st_matrix* mtxadd_(st_matrix *dest, st_matrix *a, st_matrix *b) {
    if(a->nrow != b->nrow || a->ncol != b->ncol ||
            dest->ncol != a->ncol || dest->nrow != a->nrow) {
        return NULL;
    }
    size_t i;
    size_t j;
    for(i = 0; i < a->nrow; ++i) {
        for(j = 0; j < a->ncol; ++j) {
            set(dest, i, j, get(a, i, j) + get(b, i, j));
        }
    }
    return dest;
}

st_matrix* mtxsub(st_matrix *a, st_matrix *b) {
    st_matrix *ret = malloc(sizeof(st_matrix));
    init_st_matrix(ret, a->nrow, a->ncol);
    if(!mtxadd_(ret, a, b)) {
        free_st_matrix(ret);
        free(ret);
        return NULL;
    }
    return ret;
}

st_matrix* mtxsub_(st_matrix *dest, st_matrix *a, st_matrix *b) {
    if(a->nrow != b->nrow || a->ncol != b->ncol ||
            dest->ncol != a->ncol || dest->nrow != a->nrow) {
        return NULL;
    }
    size_t i;
    size_t j;
    for(i = 0; i < a->nrow; ++i) {
        for(j = 0; j < a->ncol; ++j) {
            set(dest, i, j, get(a, i, j) - get(b, i, j));
        }
    }
    return dest;
}

st_matrix* mtxmult(st_matrix *a, st_matrix *b) {
    st_matrix *ret = malloc(sizeof(st_matrix));
    init_st_matrix(ret, a->nrow, b->ncol);
    if(!mtxmult_(ret, a, b)) {
        free_st_matrix(ret);
        free(ret);
        return NULL;
    }
    return ret;
}

st_matrix* mtxmult_(st_matrix *dest, st_matrix *a, st_matrix *b) {
    if(a->ncol != b->nrow || dest->nrow != a->nrow ||
            dest->ncol != b->ncol) {
        return NULL;
    }
    size_t i;
    size_t j;
    size_t k;
    double sum;
    for(i = 0; i < dest->nrow; ++i) {
        for(j = 0; j < dest->ncol; ++j) {
            sum = 0.0;
            for(k = 0; k < a->ncol; ++k) {
                sum += get(a, i, k) * get(b, k, j);
            }
            set(dest, i, j, sum);
        }
    }
    return dest;
}

st_matrix* mtxsmult(st_matrix *mtx, double val) {
    st_matrix *ret = malloc(sizeof(st_matrix));
    init_st_matrix(ret, mtx->nrow, mtx->ncol);
    return mtxsmult_(ret, mtx, val);
}

st_matrix* mtxsmult_(st_matrix *dest, st_matrix *src, double val) {
    size_t i;
    size_t j;
    for(i = 0; i < src->nrow; ++i) {
        for(j = 0; j < src->ncol; ++j) {
            set(dest, i, j, get(src, i, j) * val);
        }
    }
    return dest;
}

st_matrix* mtxzeros(size_t nrow, size_t ncol) {
    st_matrix *ret = malloc(sizeof(st_matrix));
    ret->mtx = calloc(nrow * ncol, sizeof(double));
    ret->nrow = nrow;
    ret->ncol = ncol;
    return ret;
}

st_matrix* mtxid(st_matrix *mtx) {
    if(mtx->nrow == mtx->ncol) {
        size_t i;
        size_t j;
        for(i = 0; i < mtx->nrow; ++i) {
            for(j = 0; j < mtx->ncol; ++j) {
                if(i == j) {
                    set(mtx, i, j, 1.0);
                } else {
                    set(mtx, i, j, 0.0);
                }
            }
        }
    }
    return mtx;
}

st_matrix* transpose(st_matrix *mtx) {
    st_matrix *ret = malloc(sizeof(st_matrix));
    init_st_matrix(ret, mtx->ncol, mtx->nrow);
    if(!transpose_(ret, mtx)) {
        free_st_matrix(ret);
        free(ret);
        return NULL;
    }
    return ret;
}

st_matrix* transpose_(st_matrix *dest, st_matrix *src) {
    if(dest->nrow != src->ncol || dest->ncol != src->nrow) {
        return NULL;
    }
    size_t i;
    size_t j;
    for(i = 0; i < src->ncol; ++i) {
        for(j = 0; j < src->nrow; ++j) {
            set(dest, i, j, get(src, j, i));
        }
    }
    return dest;
}

bool mtxeq(st_matrix *mtx1, st_matrix *mtx2) {
    if(mtx1->ncol != mtx2->ncol || mtx1->nrow != mtx2->nrow) {
        return false;
    }
    size_t i;
    size_t j;
    for(i = 0; i < mtx1->nrow; ++i) {
        for(j = 0; j < mtx1->ncol; ++j) {
            if(!deq(get(mtx1, i, j), get(mtx2, i, j))) {
                return false;
            }
        }
    }
    return true;
}

void print_st_matrix(st_matrix *mtx, size_t prec, bool lines) {
    size_t i;
    size_t j;
    size_t last = mtx->ncol - 1;
    for(i = 0; i < mtx->nrow; ++i) {
        if(lines) {
            printf("%d: ", i + 1);
        }
        for(j = 0; j < last; ++j) {
            printf("%.*f ", prec, get(mtx, i, j));
        }
        printf("%.*f\n", prec, get(mtx, i, j));
    }
}

