#ifndef _CONSTRAINT_H_
#define _CONSTRAINT_H_

#pragma once

typedef struct int_vec {
	int *get;
	size_t size;
} int_vec;

typedef struct constraint {
	int_vec *ml;
	int_vec *mnl;
} constraint;

void int_vec_init(int_vec *v, size_t size);

void int_vec_free(int_vec *v);

void int_vec_push(int_vec *v, int val);

int cmp_int(const void *a, const void *b);

void constraint_init(constraint *c, size_t mlsize, size_t mnlsize);

void constraint_free(constraint *c);

constraint** gen_constraints(int_vec *sample, size_t classc,
        size_t objc);

void print_constraints(constraint **constraints, size_t objc);

void update_constraint(constraint **c, st_matrix *memb, double in,
        double out);

#endif /* _CONSTRAINT_H_ */
