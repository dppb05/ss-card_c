#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "util.h"
#include "matrix.h"

bool load_data(char *fname, st_matrix *mtx) {
	FILE *ifile = fopen(fname, "r");
	if(!ifile) {
		return false;
	}
	size_t i;
	size_t j;
	for(i = 0; i < mtx->nrow; ++i) {
		for(j = 0; j < mtx->ncol; ++j) {
			if(fscanf(ifile, "%lf", getp(mtx, i, j)) == EOF) {
				fclose(ifile);
				return false;
			}
		}
	}
	fclose(ifile);
	return true;
}

// weightless minkowski
double minkowski(double *a, double *b, size_t size, double p) {
	size_t i;
	double ret = 0.0;
	for(i = 0; i < size; ++i) {
        ret += pow(fabs(a[i] - b[i]), p);
    }
    return pow(ret, 1.0 / p);
}

double sqdeuclid_dist(double *a, double *b, size_t size) {
    size_t i;
    double ret = 0.0;
	for(i = 0; i < size; ++i) {
        ret += pow(a[i] - b[i], 2.0);
    }
    return ret;
}

double euclid_dist(double *a, double *b, size_t size) {
    size_t i;
    double ret = 0.0;
	for(i = 0; i < size; ++i) {
        ret += pow(a[i] - b[i], 2.0);
    }
    return pow(ret, 0.5);
}

void print_mtx_d_(double **matrix, size_t nrow, size_t ncol) {
	size_t i;
	size_t j;
	for(i = 0; i < nrow; ++i) {
		for(j = 0; j < ncol - 1; ++j) {
			printf("%lf ", matrix[i][j]);
		}
		printf("%lf\n", matrix[i][j]);
	}
}

void print_mtx_d(double *mtx, size_t nrow, size_t ncol) {
    size_t i;
    size_t j;
    size_t last = ncol - 1;
    for(i = 0; i < nrow; ++i) {
        for(j = 0; j < last; ++j) {
            printf("%lf ", mtx[i * ncol + j]);
        }
        printf("%lf\n", mtx[i * ncol + j]);
    }
}

void fprint_mtx_d(FILE *file, double **matrix, size_t nrow,
					size_t ncol) {
	size_t i;
	size_t j;
	for(i = 0; i < nrow; ++i) {
		for(j = 0; j < ncol - 1; ++j) {
			fprintf(file, "%.4lf ", matrix[i][j]);
		}
		fprintf(file, "%.4lf\n", matrix[i][j]);
	}
}

void print_mtx_size_t(size_t **matrix, size_t nrow, size_t ncol) {
	size_t i;
	size_t j;
	for(i = 0; i < nrow; ++i) {
		for(j = 0; j < ncol - 1; ++j) {
			printf("%u ", matrix[i][j]);
		}
		printf("%u\n", matrix[i][j]);
	}
}

void print_mtx_int(int *mtx, size_t nrow, size_t ncol) {
    size_t i;
    size_t j;
    size_t last = ncol - 1;
    for(i = 0; i < nrow; ++i) {
        for(j = 0; j < last; ++j) {
            printf("%d ", mtx[i * ncol + j]);
        }
        printf("%d\n", mtx[i * ncol + j]);
    }
}

bool deq(double a, double b) {
//    return fabs(a - b) < FPOINT_OFFSET;
    return (a < (b + FPOINT_OFFSET) && a > (b - FPOINT_OFFSET));
}

bool dgt(double a, double b) {
    return a > (b + FPOINT_OFFSET);
}

bool dlt(double a, double b) {
    return a < (b - FPOINT_OFFSET);
}

int cmpint(const void *a, const void *b) {
    return ( *(int*)a - *(int*)b );
}

//void* binsearch(const void *key, const void *base, size_t num,
//        size_t size, int (*cmp) (const void *key, const void *elt)) {
//    size_t start = 0;
//    size_t end = num;
//    int result;
//    while(start < end) {
//        size_t mid = start + (end - start) / 2;
//        result = cmp(key, base + mid * size);
//        if(result < 0) {
//            end = mid;
//        } else if(result > 0) {
//            start = mid + 1;
//        } else {
//            return (void *) base + mid * size;
//        }
//    }
//    return NULL;
//}

//int unique(int *vec, size_t size) {
//    int uniq[size];
//    size_t i;
//    int ret = 0;
//    int *cur;
//    for(i = 0; i < size; ++i) {
//        cur = bsearch(vec[i], uniq, sizeof(int), cmpint);
//        if(!cur) {
//            ++ret;
//        }
//        if(!exist[vec[i]]) {
//            ++ret;
//            exist[vec[i]] = true;
//        }
//    }
//    return ret;
//}

int max(int *vec, size_t size) {
    if(!size) {
        return 0;
    }
    int ret = vec[0];
    size_t i;
    for(i = 1; i < size; ++i) {
        if(vec[i] > ret) {
            ret = vec[i];
        }
    }
    return ret;
}

// Prints a header padded with '-' having 'str' in the center.
// Params:
//  str - an string to be print as header.
//  size - length of the header, this has to be at least strlen(str)
void print_header(char *str, size_t size) {
    size_t str_size = strlen(str);
    size_t buf_size = size;
    if(buf_size < str_size) {
        buf_size = str_size;
    }
    char buffer[buf_size];
    size_t i;
    size_t last = (buf_size / 2) - (str_size / 2);
    for(i = 0; i < last; ++i) {
        buffer[i] = '-';
    }
    size_t j;
    for(j = 0; j < str_size; ++i, ++j) {
        buffer[i] = str[j];
    }
    for(; i < buf_size; ++i) {
        buffer[i] = '-';
    }
    buffer[i] = '\0';
    printf("\n%s\n", buffer);
}

void mtxcpy_d(double **destination, double **source, size_t nrow,
        size_t ncol) {
    size_t i;
    for(i = 0; i < nrow; ++i) {
        memcpy(destination[i], source[i], sizeof(double) * ncol);
    }
}

void mtxcpy_size_t(size_t **destination, size_t **source, size_t nrow,
        size_t ncol) {
    size_t i;
    for(i = 0; i < nrow; ++i) {
        memcpy(destination[i], source[i], sizeof(size_t) * ncol);
    }
}

