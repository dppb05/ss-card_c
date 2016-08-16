#ifndef _STEX_H_
#define _STEX_H_

#pragma once

#include "matrix.h"

typedef struct st_silhouet {
    size_t objc;
    size_t clustc;
    double *objsil;
    double *clustsil;
    double avgsil;
} silhouet;

void remap(int *labels, size_t size, int *factors);

int* defuz(st_matrix *fuzmtx);

st_matrix* confusion(int *labels, int *pred, size_t size);

double partcoef(st_matrix *fuzmtx);

double modpcoef(st_matrix *fuzmtx);

double partent(st_matrix *fuzmtx);

double corand(int *labels, int *pred, size_t size);

double avg_intra_dist(st_matrix *fuzmtx, st_matrix *dist, double mfuz);

silhouet* crispsil(st_matrix *groups, st_matrix *dmatrix);

silhouet* simplesil(int *labels, st_matrix *cent_dist);

silhouet* fuzzysil(silhouet *sil, st_matrix *groups, st_matrix *memb,
                    double alpha);

st_matrix* asgroups(int *labels, size_t size, size_t card);

//void print_groups(int *labels, size_t size, size_t card);
void print_groups(st_matrix *groups);

silhouet* avg_silhouet(silhouet *s1, silhouet *s2);

void free_silhouet(silhouet *s);

void print_silhouet(silhouet *s);

#endif /* _STEX_H_ */
