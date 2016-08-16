// TODO:
//  - handle case where there are no constraints (sample_perc = 0.0)
//  as it might cause errors (verify)
//  - use only one object partition group from labels: exclude the
//  old one copied from SetMedoids-MLMNL-P and adapt the sampling
//  accordingly
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include "util.h"
#include "matrix.h"
#include "stex.h"

#define BUFF_SIZE 1024
#define HEADER_SIZE 51

typedef struct int_vec {
	int *get;
	size_t size;
} int_vec;

void int_vec_init(int_vec *v, size_t size) {
	v->get = malloc(sizeof(int) * size);
	v->size = 0;
}

void int_vec_free(int_vec *v) {
	free(v->get);
}

void int_vec_push(int_vec *v, int val) {
	v->get[v->size++] = val;
}

typedef struct constraint {
	int_vec *ml;
	int_vec *mnl;
} constraint;

void constraint_init(constraint *c, size_t mlsize, size_t mnlsize) {
    c->ml = malloc(sizeof(int_vec));
	int_vec_init(c->ml, mlsize);
    c->mnl = malloc(sizeof(int_vec));
	int_vec_init(c->mnl, mnlsize);
}

void constraint_free(constraint *c) {
	int_vec_free(c->ml);
    free(c->ml);
	int_vec_free(c->mnl);
    free(c->mnl);
	free(c);
}

bool verbose;
double epsilon;
double qexp;
double qexpval;
double beta;
double alpha;
int objc;
int dmatrixc;
int clustc;
int max_iter;
st_matrix memb;
st_matrix *dmatrix;
st_matrix *global_dmatrix;
st_matrix *membvec;
st_matrix dists;
st_matrix weights;
int_vec *class;
int classc;
int_vec *sample;
constraint **constraints;
size_t constsc;
st_matrix a_val;
//st_matrix c_val;
//double *c_val_obj;
// 'cluster_sum' stores the following for a given cluster 'i':
// SUM_j=1^N SUM_k=1^N u_ij^2 * u_ik^2 * SUM_s=1^S w_is^q * r_jk^s
double *cluster_sum; 
// 'sqd_memb_sum' stores the following for a given cluster 'i':
// SUM_k=1^N u_ik^2
double *sqd_memb_sum; 

void init_memb() {
    size_t i;
    size_t k;
    double sum;
    double val;
    for(i = 0; i < objc; ++i) {
        sum = 0.0;
        for(k = 0; k < clustc; ++k) {
            val = rand();
            sum += val;
            set(&memb, i, k, val);
        }
        for(k = 0; k < clustc; ++k) {
            set(&memb, i, k, get(&memb, i, k) / sum);
        }
    }
}

void print_memb(st_matrix *memb) {
	printf("Membership:\n");
	size_t i;
	size_t k;
	double sum;
    double val;
	for(i = 0; i < objc; ++i) {
		printf("%u: ", i);
		sum = 0.0;
		for(k = 0; k < clustc; ++k) {
            val = get(memb, i, k);
			printf("%lf ", val);
			sum += val;
		}
		printf("[%lf]", sum);
		if(!deq(sum, 1.0)) {
			printf("*\n");
		} else {
			printf("\n");
		}
	}
}

void init_weights() {
    size_t k;
    size_t j;
    double val = 1.0 / dmatrixc;
    for(k = 0; k < clustc; ++k) {
        for(j = 0; j < dmatrixc; ++j) {
            set(&weights, k, j, val);
        }
    }
}

void print_weights(st_matrix *weights) {
	printf("Weights:\n");
	size_t j;
	size_t k;
	double sum;
    double val;
	for(k = 0; k < clustc; ++k) {
		sum = 0.0;
		for(j = 0; j < dmatrixc; ++j) {
            val = get(weights, k, j);
			if(dlt(val, 0.0)) {
				printf("!");
			}
			printf("%lf ", val);
			sum += val;
		}
		printf("[%lf]", sum);
		if(!deq(sum, 1.0)) {
			printf(" =/= 1.0?\n");
		} else {
			printf("\n");
		}
	}
}

double adequacy() {
    size_t h;
    size_t i;
    size_t j;
    size_t k;
    double sum_num;
    double sum_den;
    double sumd;
    double adeq = 0.0;
    for(k = 0; k < clustc; ++k) {
        sum_num = 0.0;
        sum_den = 0.0;
        for(i = 0; i < objc; ++i) {
            for(h = 0; h < objc; ++h) {
                sumd = 0.0;
                for(j = 0; j < dmatrixc; ++j) {
                    sumd += pow(get(&weights, k, j), qexp) *
                        get(&dmatrix[j], i, h);
                }
                sum_num += pow(get(&memb, i, k), 2.0) *
                    pow(get(&memb, h, k), 2.0) * sumd;
            }
            sum_den += pow(get(&memb, i, k), 2.0);
        }
        adeq += (sum_num / (2.0 * sum_den));
    }
    size_t m;
    size_t r;
    size_t s;
    size_t obj;
    double constr_adeq = 0.0;
    for(i = 0; i < objc; ++i) {
        if(constraints[i]) {
            for(m = 0; m < constraints[i]->ml->size; ++m) {
                obj = constraints[i]->ml->get[m];
                for(r = 0; r < clustc; ++r) {
                    for(s = 0; s < clustc; ++s) {
                        if(s != r) {
                            constr_adeq += get(&memb, i, r) *
                                get(&memb, obj, s);
                        }
                    }
                }
            }
            for(m = 0; m < constraints[i]->mnl->size; ++m) {
                obj = constraints[i]->mnl->get[m];
                for(r = 0; r < clustc; ++r) {
                    constr_adeq += get(&memb, i, r) *
                        get(&memb, obj, r);
                }
            }
        }
    }
    printf("adeq: %lf\nconstr_adeq: %lf\n", adeq, constr_adeq);
    return adeq + alpha * constr_adeq;
}

void global_dissim() {
    size_t h;
    size_t i;
    size_t j;
    size_t k;
    double sum;
    for(k = 0; k < clustc; ++k) {
        for(i = 0; i < objc; ++i) {
            for(h = 0; h < objc; ++h) {
                sum = 0.0;
                for(j = 0; j < dmatrixc; ++j) {
                    sum += pow(get(&weights, k, j), qexp) *
                        get(&dmatrix[j], i, h);
                }
                if(i != h) {
                    sum += beta;
                }
                set(&global_dmatrix[k], i, h, sum);
            }
        }
    }
}

void compute_membvec() {
    size_t k;
    size_t i;
    double sum_den;
    double val;
    for(k = 0; k < clustc; ++k) {
        sum_den = 0.0;
        for(i = 0; i < objc; ++i) {
            val = pow(get(&memb, i, k), 2.0);
            set(&membvec[k], i, 0, val);
            sum_den += val; 
        }
        for(i = 0; i < objc; ++i) {
            set(&membvec[k], i, 0, get(&membvec[k], i, 0) / sum_den);
        }
    }
}

bool compute_dists() {
    size_t h;
    size_t i;
    size_t j;
    size_t k;
    double val;
    double sumd;
    double aval_inv_sum;
    double sum_num;
    double sum_den;
    bool hasneg = false;
    for(k = 0; k < clustc; ++k) {
        cluster_sum[k] = 0.0;
        sqd_memb_sum[k] = 0.0;
        for(i = 0; i < objc; ++i) {
            for(h = 0; h < objc; ++h) {
                sumd = 0.0;
                for(j = 0; j < dmatrixc; ++j) {
                    sumd += pow(get(&weights, k, j), qexp) *
                        get(&dmatrix[j], i, h);
                }
                cluster_sum[k] += pow(get(&memb, i, k), 2.0) *
                    pow(get(&memb, h, k), 2.0) * sumd;
            }
            sqd_memb_sum[k] += pow(get(&memb, i, k), 2.0);
        }
        for(i = 0; i < objc; ++i) {
            sum_num = 0.0;
            for(h = 0; h < objc; ++h) {
                sumd = 0.0;
                for(j = 0; j < dmatrixc; ++j) {
                    sumd += pow(get(&weights, k, j), qexp) *
                        get(&dmatrix[j], i, h);
                }
                sum_num += pow(get(&memb, h, k), 2.0) * sumd;
            }
            val = ((2.0 * sum_num) / sqd_memb_sum[k]) -
                    (cluster_sum[k] / pow(sqd_memb_sum[k], 2.0));
            set(&a_val, i, k, val);
            set(&dists, k, i, val / 2.0);
            if(!hasneg && val < 0.0) {
                hasneg = true;
            }
        }
    }
    return hasneg;
}

double compute_deltabeta() {
    size_t i;
    size_t k;
    double val;
    double deltabeta;
    bool first = true;
    double idcol[objc];
    for(i = 0; i < objc; ++i) {
        idcol[i] = 0.0;
    }
    for(i = 0; i < objc; ++i) {
        idcol[i] = 1.0;
        for(k = 0; k < clustc; ++k) {
            val = pow(euclid_dist(membvec[k].mtx, idcol, objc), 2.0);
            if(val != 0.0) {
                val = (-2.0 * get(&dists, k, i)) / val;
                if(first || val > deltabeta) {
                    deltabeta = val;
                    first = false;
                }
            }
        }
        idcol[i] = 0.0;
    }
    return deltabeta;
}

bool adjust_dists() {
    printf("Adjusting distances...\n");
    int i;
    size_t k;
    double deltabeta = compute_deltabeta();
    printf("deltabeta: %.15lf\n", deltabeta);
    beta += deltabeta;
    printf("beta: %.15lf\n", beta);
    deltabeta /= 2.0;
    double idcol[objc];
    for(i = 0; i < objc; ++i) {
        idcol[i] = 0.0;
    }
    bool hasneg = false;
    for(i = 0; i < objc; ++i) {
        for(k = 0; k < clustc; ++k) {
            idcol[i] = 1.0;
            set(&dists, k, i, get(&dists, k, i) + deltabeta *
                    sqdeuclid_dist(membvec[k].mtx, idcol, objc));
            idcol[i] = 0.0;
            if(!hasneg && dlt(get(&dists, k, i), 0.0)) {
                hasneg = true;
            }
        }
    }
    return hasneg;
}

void adjust_a_val() {
    size_t i;
    size_t k;
    for(i = 0; i < objc; ++i) {
        for(k = 0; k < clustc; ++k) {
            set(&a_val, i, k, 2.0 * get(&dists, k, i));
        }
    }
}

void update_memb() {
    size_t c;
    size_t i;
    size_t k;
    double val;
    int zerovalc;
    for(i = 0; i < objc; ++i) {
        zerovalc = 0;
        for(k = 0; k < clustc; ++k) {
            if(!(get(&dists, k, i) > 0.0)) {
                ++zerovalc;
            }
        }
        if(zerovalc) {
            printf("Msg: there is at least one zero val for d[%d]."
                    "\n", i);
            val = 1.0 / ((double) zerovalc);
            for(k = 0; k < clustc; ++k) {
                if(get(&dists, k, i) > 0.0) {
                    set(&memb, i, k, 0.0);
                } else {
                    set(&memb, i, k, val);
                }
            }
        } else {
            for(k = 0; k < clustc; ++k) {
                val = 0.0;
                for(c = 0; c < clustc; ++c) {
                    val += pow(get(&dists, k, i) / get(&dists, c, i),
                            1.0);
                }
                set(&memb, i, k, 1.0 / val);
            }
        }
    }
}

void constr_update_memb_old() {
    size_t c;
    size_t e;
    size_t h;
    size_t i;
    size_t j;
    size_t k;
    size_t m;
    double val;
    double sumd;
    double a_val[clustc];
    double a_val_inv_sum;
    double c_val[clustc];
    double c_val_sum;
    double sum_num1;
    double sum_num2;
    double sum_den;
    double memb_rfcm;
    double memb_constr;
    bool normalize;
    for(i = 0; i < objc; ++i) {
        sum_num1 = 0.0;
        sum_num2 = 0.0;
        sum_den = 0.0;
        c_val_sum = 0.0;
        a_val_inv_sum = 0.0;
        for(k = 0; k < clustc; ++k) {
            for(h = 0; h < objc; ++h) {
                for(e = 0; e < objc; ++e) {
                    sumd = 0.0;
                    for(j = 0; j < dmatrixc; ++j) {
                        sumd += pow(get(&weights, k, j), qexp) *
                            get(&dmatrix[j], i, h);
                    }
                    sum_num2 += pow(get(&memb, h, k), 2.0) *
                        pow(get(&memb, e, k), 2.0) * sumd;
                }
                sumd = 0.0;
                for(j = 0; j < dmatrixc; ++j) {
                    sumd += pow(get(&weights, k, j), qexp) *
                        get(&dmatrix[j], i, h);
                }
                val = pow(get(&memb, h, k), 2.0);
                sum_num1 += val * sumd;
                sum_den += val;
            }
            a_val[k] = ((2.0 * sum_num1) / sum_den) -
                sum_num2 / pow(sum_den, 2.0);
            c_val[k] = 0.0;
            if(constraints[i]) {
                for(m = 0; m < constraints[i]->ml->size; ++m) {
                    for(c = 0; c < clustc; ++c) {
                        c_val[k] +=
                            get(&memb, constraints[i]->ml->get[m], c);
                    }
                }
                for(m = 0; m < constraints[i]->mnl->size; ++m) {
                    c_val[k] +=
                        get(&memb, constraints[i]->mnl->get[m], k);
                }
            }
            c_val_sum += c_val[k] / a_val[k];
            a_val_inv_sum += 1.0 / a_val[k];
        }
        c_val_sum /= a_val_inv_sum;
        normalize = false;
        sum_den = 0.0;
        for(k = 0; k < clustc; ++k) {
            memb_rfcm = 0.0;
            for(c = 0; c < clustc; ++c) {
                memb_rfcm += a_val[k] / a_val[c];
            }
            memb_rfcm = 1.0 / memb_rfcm;
            memb_constr = (alpha / a_val[k]) * (c_val_sum - c_val[k]);
            val = memb_rfcm + memb_constr;
            if(val < 0.0) {
                normalize = true;
                val = 0.0;
            } else if(val > 1.0) {
                normalize = true;
                val = 1.0;
            } else {
                sum_den += val;
            }
            set(&memb, i, k, val);
        }
        if(normalize) {
            printf("Msg: performing clipping on object %d.\n", i);
            for(k = 0; k < clustc; ++k) {
                val = get(&memb, i, k);
                if(val < 1.0 && val > 0.0) {
                    set(&memb, i, k, val / sum_den);
                }
            }
        }
    }
}

void constr_update_memb() {
    size_t c;
    size_t h;
    size_t i;
    size_t k;
    size_t obj;
    double val;
    double memb_sum;
    double memb_rfcm;
    double memb_constr;
    double c_val[clustc];
    double c_val_obj;
    double a_val_inv_sum;
    bool clip;
    for(i = 0; i < objc; ++i) {
        if(constraints[i]) {
            a_val_inv_sum = 0.0;
            c_val_obj = 0.0;
            for(k = 0; k < clustc; ++k) {
                c_val[k] = 0.0;
                for(h = 0; h < constraints[i]->ml->size; ++h) {
                    obj = constraints[i]->ml->get[h];
                    for(c = 0; c < clustc; ++c) {
                        if(c != k) {
                            c_val[k] += get(&memb, obj, c);
                        }
                    }
                }
                for(h = 0; h < constraints[i]->mnl->size; ++h) {
                    obj = constraints[i]->mnl->get[h];
                    c_val[k] += get(&memb, obj, k);
                }
                c_val_obj += c_val[k] / get(&a_val, i, k);
                a_val_inv_sum += 1.0 / get(&a_val, i, k);
            }
            c_val_obj /= a_val_inv_sum;
        }
        memb_sum = 0.0;
        clip = false;
        for(k = 0; k < clustc; ++k) {
            memb_rfcm = 0.0;
            for(c = 0; c < clustc; ++c) {
                memb_rfcm += get(&a_val, i, k) / get(&a_val, i, c);
            }
            if(constraints[i]) {
                val = (1.0 / memb_rfcm) +
                        (alpha / get(&a_val, i, k)) *
                        (c_val_obj - c_val[k]);
            } else {
                val = (1.0 / memb_rfcm);
            }
            if(val < 0.0) {
                val = 0.0;
                clip = true;
            } else if(val > 1.0) {
                val = 1.0;
                clip = true;
            }
            memb_sum += val;
            set(&memb, i, k, val);
        }
        if(clip) {
            printf("Msg: clipping membership for object %d\n", i);
            for(k = 0; k < clustc; ++k) {
                val = get(&memb, i, k);
                if(val > 0.0 && val <= 1.0) {
                    set(&memb, i, k, val / memb_sum);
                }
            }
        }
    }
}

void update_weights() {
    size_t h;
    size_t i;
    size_t j;
    size_t k;
    size_t p;
    double dispersion[dmatrixc];
    double val;
    int zeroc;
    for(k = 0; k < clustc; ++k) {
        zeroc = 0;
        printf("Dispersions:\n");
        for(j = 0; j < dmatrixc; ++j) {
            val = 0.0;
            for(i = 0; i < objc; ++i) {
                for(h = 0; h < objc; ++h) {
                    val += pow(get(&memb, i, k), mfuz) *
                        pow(get(&memb, h, k), mfuz) *
                        get(&dmatrix[j], i, h);
                }
            }
            if(!val) {
                ++zeroc;
            }
            dispersion[j] = val;
            printf("%.15lf ", val);
        }
        printf("\n");
        if(zeroc) {
            printf("Msg: at least one dispersion is zero for cluster"
                    " %d\n", k);
            val = 1.0 / zeroc;
            for(j = 0; j < dmatrixc; ++j) {
                if(!dispersion[j]) {
                    set(&weights, k, j, val);
                } else {
                    set(&weights, k, j, 0.0);
                }
            }
        } else {
            for(j = 0; j < dmatrixc; ++j) {
                val = 0.0;
                for(p = 0; p < dmatrixc; ++p) {
                    val += pow(dispersion[j] / dispersion[p],
                            qexpval);
                }
                if(!val) {
                        printf("Warn: division by zero\n");
                }
                set(&weights, k, j, 1.0 / val);
            }
        }
    }
}

void update_alpha() {
    size_t c;
    size_t h;
    size_t i;
//    size_t j;
    size_t k;
    double sum_num = 0.0;
//    double sum_memb;
//    double sum_obj;
//    double sumd;
    for(k = 0; k < clustc; ++k) {
        sum_num += cluster_sum[k] / (2.0 * sqd_memb_sum[k]);
//        sum_obj = 0.0;
//        sum_memb = 0.0;
//        for(i = 0; i < objc; ++i) {
//            for(h = 0; h < objc; ++h) {
//                sumd = 0.0;
//                for(j = 0; j < dmatrixc; ++j) {
//                    sumd += pow(get(&weights, k, j), qexp) *
//                            get(&dmatrix[j], i, h);
//                }
//                sum_obj += pow(get(&memb, i, k), 2.0) *
//                            pow(get(&memb, h, k), 2.0) * sumd;
//            }
//            sum_memb += pow(get(&memb, i, k), 2.0);
//        }
//        sum_num += sum_obj / (2 * sum_memb);
    }
    double sum_den = 0.0;
    size_t obj;
    for(i = 0; i < objc; ++i) {
        if(constraints[i]) {
            for(h = 0; h < constraints[i]->ml->size; ++h) {
                obj = constraints[i]->ml->get[h];
                for(k = 0; k < clustc; ++k) {
                    for(c = 0; c < clustc; ++c) {
                        if(c != k) {
                            sum_den += get(&memb, i, k) *
                                        get(&memb, obj, c);
                        }
                    }
                }
            }
            for(h = 0; h < constraints[i]->mnl->size; ++h) {
                obj = constraints[i]->mnl->get[h];
                for(k = 0; k < clustc; ++k) {
                    sum_den += get(&memb, i, k) * get(&memb, obj, k);
                }
            }
        }
    }
    printf("num: %lf\nden: %lf\n", sum_num, sum_den);
    alpha = sum_num / sum_den;
}

double run() {
    printf("Initialization.\n");
    init_memb();
    if(verbose) print_memb(&memb);
    init_weights();
    if(verbose) print_weights(&weights);
    compute_dists(); // pre-compute 'cluster_sum' and 'sqd_memb_sum'
    update_alpha();
    printf("Alpha: %.10f\n", alpha);
    beta = 0.0;
    double adeq = adequacy();
    printf("Adequacy: %.15lf\n", adeq);
    double prev_iter_adeq;
    double adeq_diff;
    size_t iter = 1;
    st_matrix prev_memb;
    init_st_matrix(&prev_memb, objc, clustc);
    do {
        printf("Iteration %d:\n", iter);
        prev_iter_adeq = adeq;
//        global_dissim();
//        compute_membvec();
        if(compute_dists()) {
            do {
                if(verbose) {
                    printf("Distances:\n");
                    print_st_matrix(&dists, 10, true);
                }
            } while(adjust_dists());
        }
        if(verbose) {
            printf("Distances:\n");
            print_st_matrix(&dists, 10, true);
        }
        adjust_a_val();
        update_alpha();
        printf("Alpha: %.10f\n", alpha);
        mtxcpy(&prev_memb, &memb);
        constr_update_memb();
//        update_memb();
        if(verbose) print_memb(&memb);
        update_weights();
        if(verbose) print_weights(&weights);
        adeq = adequacy();
        printf("Adequacy: %.15lf\n", adeq);
        adeq_diff = prev_iter_adeq - adeq;
        if(adeq_diff < 0.0) {
            adeq_diff = fabs(adeq_diff);
            printf("Warn: previous iteration adequacy is greater "
                    "than current (%.15lf).\n", adeq_diff);
        }
        if(adeq_diff < epsilon) {
            printf("Adequacy difference threshold reached (%.15lf)."
                    "\n", adeq_diff);
            break;
        }
        if(++iter > max_iter) {
            printf("Maximum number of iterations reached.\n");
            break;
        }
    } while(true);
    free_st_matrix(&prev_memb);
    printf("Beta: %.15lf\n", beta);
    return adeq;
}

void aggregate_dmatrices(st_matrix *dest, st_matrix *weights) {
    size_t h;
    size_t i;
    size_t j;
    size_t k;
    double val;
    for(i = 0; i < objc; ++i) {
        for(h = 0; h < objc; ++h) {
            val = 0.0;
            for(k = 0; k < clustc; ++k) {
                for(j = 0; j < dmatrixc; ++j) {
                    val += pow(get(weights, k, j), qexp) *
                        get(&dmatrix[j], i, h);
                }
            }
            set(dest, i, h, val);
        }
    }
}

void gen_sample(size_t size) {
	printf("sample size: %d\n", size);
	sample = malloc(sizeof(int_vec) * classc);
	size_t per_class = size / classc;
	constsc = per_class * classc;
	int pos;
	size_t i;
	size_t k;
	for(k = 0; k < classc; ++k) {
		int_vec_init(&sample[k], per_class);
		bool chosen[class[k].size];
        for(i = 0; i < class[k].size; ++i) {
            chosen[i] = false;
        }
		for(i = 0; i < per_class; ++i) {
			do {
				pos = rand() % class[k].size;
			} while(chosen[pos]);
			chosen[pos] = true;
			int_vec_push(&sample[k], class[k].get[pos]);
		}
	}
}

void print_sample() {
	size_t i;
	size_t k;
	for(k = 0; k < classc; ++k) {
		printf("Sample %d (%d members):\n", k, sample[k].size);
		for(i = 0; i < sample[k].size; ++i) {
			printf("%u ", sample[k].get[i]);
		}
		printf("\n");
	}
}

void gen_constraints() {
	constraints = calloc(objc, sizeof(constraint *));
	size_t i;
	size_t e;
	size_t h;
	size_t k;
	size_t obj;
	size_t obj2;
	for(k = 0; k < classc; ++k) {
		for(i = 0; i < sample[k].size; ++i) {
			obj = sample[k].get[i];
            constraints[obj] = malloc(sizeof(constraint));
			constraint_init(constraints[obj], sample[k].size,
							constsc - sample[k].size);
			for(h = 0; h < classc; ++h) {
				for(e = 0; e < sample[h].size; ++e) {
					obj2 = sample[h].get[e];
					if(obj != obj2) {
						if(h == k) {
							int_vec_push(constraints[obj]->ml, obj2);
						} else {
							int_vec_push(constraints[obj]->mnl, obj2);
						}
					}
				}
			}
		}
	}
}

void print_constraints() {
    printf("Constraints:\n");
    size_t e;
    size_t i;
    size_t k;
    size_t obj;
	for(k = 0; k < classc; ++k) {
		for(i = 0; i < sample[k].size; ++i) {
			obj = sample[k].get[i];
            printf("Obj %d:\n", obj);
            printf("ML:");
            for(e = 0; e < constraints[obj]->ml->size; ++e) {
                printf(" %d", constraints[obj]->ml->get[e]);
            }
            printf("\n");
            printf("MNL:");
            for(e = 0; e < constraints[obj]->mnl->size; ++e) {
                printf(" %d", constraints[obj]->mnl->get[e]);
            }
            printf("\n");
		}
	}
}

void print_class() {
	size_t i;
	size_t k;
	for(k = 0; k < classc; ++k) {
		printf("Class %d (%d members):\n", k, class[k].size);
		for(i = 0; i < class[k].size; ++i) {
			printf("%u ", class[k].get[i]);
		}
		printf("\n");
	}
}

int main(int argc, char **argv) {
    verbose = true;
    FILE *cfgfile = fopen(argv[1], "r");
    if(!cfgfile) {
        printf("Error: could not open config file.\n");
        return 1;
    }
    fscanf(cfgfile, "%d", &objc);
    if(objc <= 0) {
        printf("Error: objc <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    // reading labels
    int labels[objc];
    fscanf(cfgfile, "%d", &classc);
    size_t i;
    for(i = 0; i < objc; ++i) {
        fscanf(cfgfile, "%d", &labels[i]);
    }
    // reading labels end
    fscanf(cfgfile, "%d", &dmatrixc);
    if(dmatrixc <= 0) {
        printf("Error: dmatrixc <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    char filenames[dmatrixc][BUFF_SIZE];
    size_t j;
    for(j = 0; j < dmatrixc; ++j) {
        fscanf(cfgfile, "%s", filenames[j]);
    }
    char outfilename[BUFF_SIZE];
    fscanf(cfgfile, "%s", outfilename);
    fscanf(cfgfile, "%d", &clustc);
    if(clustc <= 0) {
        printf("Error: clustc <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    int insts;
    fscanf(cfgfile, "%d", &insts);
    if(insts <= 0) {
        printf("Error: insts <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%d", &max_iter);
    if(insts <= 0) {
        printf("Error: max_iter <= 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%lf", &epsilon);
    if(dlt(epsilon, 0.0)) {
        printf("Error: epsilon < 0.\n");
        fclose(cfgfile);
        return 1;
    }
    fscanf(cfgfile, "%lf", &qexp);
    if(dlt(qexp, 1.0)) {
        printf("Error: qexp < 1.0.\n");
        fclose(cfgfile);
        return 1;
    }
    double sample_perc;
    fscanf(cfgfile, "%lf", &sample_perc);
    if(dlt(sample_perc, 0.0)) {
        printf("Error: sample_perc < 0.\n");
        return 2;
    }
    fclose(cfgfile);
    freopen(outfilename, "w", stdout);
    printf("###Configuration summary:###\n");
    printf("Number of objects: %d\n", objc);
    printf("Number of clusters: %d\n", clustc);
    printf("Number of instances: %d\n", insts);
    printf("Maximum interations: %d\n", max_iter);
    printf("Parameter q: %lf\n", qexp);
    printf("Sample perc: %lf\n", sample_perc);
    printf("############################\n");
    st_matrix best_memb;
    st_matrix best_dists;
    st_matrix best_weights;
    // memory allocation start
    dmatrix = malloc(sizeof(st_matrix) * dmatrixc);
    for(j = 0; j < dmatrixc; ++j) {
        init_st_matrix(&dmatrix[j], objc, objc);
    }
    init_st_matrix(&memb, objc, clustc);
    init_st_matrix(&best_memb, objc, clustc);
    size_t k;
    membvec = malloc(sizeof(st_matrix) * clustc);
    global_dmatrix = malloc(sizeof(st_matrix) * clustc);
    for(k = 0; k < clustc; ++k) {
        init_st_matrix(&membvec[k], objc, 1);
        init_st_matrix(&global_dmatrix[k], objc, objc);
    }
    init_st_matrix(&dists, clustc, objc);
    init_st_matrix(&best_dists, clustc, objc);
    init_st_matrix(&weights, clustc, dmatrixc);
    init_st_matrix(&best_weights, clustc, dmatrixc);
    class = malloc(sizeof(int_vec) * classc);
    for(i = 0; i < classc; ++i) {
        int_vec_init(&class[i], objc);
    }
    init_st_matrix(&a_val, objc, clustc);
//    init_st_matrix(&c_val, objc, clustc);
//    c_val_obj = malloc(sizeof(double) * objc);
    cluster_sum = malloc(sizeof(double) * clustc);
    sqd_memb_sum = malloc(sizeof(double) * objc);
	// Allocating memory end
    // Loading labels
    for(i = 0; i < objc; ++i) {
        int_vec_push(&class[labels[i]], i);
    }
	print_class();
    // Loading matrices
    for(j = 0; j < dmatrixc; ++j) {
        if(!load_data(filenames[j], &dmatrix[j])) {
            printf("Error: could not load matrix file.\n");
            goto END;
        }
    }
    double mfuz = 1.6;
    qexpval = 1.0 / (qexp - 1.0);
    double avg_partcoef;
    double avg_modpcoef;
    double avg_partent;
    double avg_aid;
    st_matrix dists_t;
    init_st_matrix(&dists_t, dists.ncol, dists.nrow);
    st_matrix agg_dmatrix;
    init_st_matrix(&agg_dmatrix, objc, objc);
    silhouet *csil;
    silhouet *fsil;
    silhouet *ssil;
    silhouet *avg_csil;
    silhouet *avg_fsil;
    silhouet *avg_ssil;
    int *pred;
    st_matrix *groups;
    srand(time(NULL));
    size_t best_inst;
    double best_inst_adeq;
    double cur_inst_adeq;
	gen_sample(sample_perc * objc);
	print_sample();
	gen_constraints();
    print_constraints();
    for(i = 1; i <= insts; ++i) {
        printf("Instance %d:\n", i);
        cur_inst_adeq = run();
        pred = defuz(&memb);
        groups = asgroups(pred, objc, classc);
        transpose_(&dists_t, &dists);
        aggregate_dmatrices(&agg_dmatrix, &weights);
        csil = crispsil(groups, &agg_dmatrix);
        fsil = fuzzysil(csil, groups, &memb, mfuz);
        ssil = simplesil(pred, &dists_t);
        if(i == 1) {
            avg_partcoef = partcoef(&memb);
            avg_modpcoef = modpcoef(&memb);
            avg_partent = partent(&memb);
            avg_aid = avg_intra_dist(&memb, &dists_t, mfuz);
            avg_csil = csil;
            avg_fsil = fsil;
            avg_ssil = ssil;
        } else {
            avg_partcoef = (avg_partcoef + partcoef(&memb)) / 2.0;
            avg_modpcoef = (avg_modpcoef + modpcoef(&memb)) / 2.0;
            avg_partent = (avg_partent + partent(&memb)) / 2.0;
            avg_aid = (avg_aid +
                        avg_intra_dist(&memb, &dists_t, mfuz)) / 2.0;
            avg_silhouet(avg_csil, csil);
            avg_silhouet(avg_fsil, fsil);
            avg_silhouet(avg_ssil, ssil);
            free_silhouet(csil);
            free(csil);
            free_silhouet(fsil);
            free(fsil);
            free_silhouet(ssil);
            free(ssil);
        }
        free(pred);
        free_st_matrix(groups);
        free(groups);
        if(i == 1 || cur_inst_adeq < best_inst_adeq) {
            mtxcpy(&best_memb, &memb);
            mtxcpy(&best_dists, &dists);
            mtxcpy(&best_weights, &weights);
            best_inst_adeq = cur_inst_adeq;
            best_inst = i;
        }
    }
	printf("\n");
    printf("Best adequacy %.15lf on instance %d.\n", best_inst_adeq,
            best_inst);
    printf("\n");
    print_memb(&best_memb);
    print_weights(&best_weights);

    pred = defuz(&best_memb);
    // TODO: use 'asgroups' to make the constraints
    groups = asgroups(pred, objc, classc);
    print_header("Partitions", HEADER_SIZE);
    print_groups(groups);

    print_header("Average indexes", HEADER_SIZE);
    printf("\nPartition coefficient: %.10lf\n", avg_partcoef);
    printf("Modified partition coefficient: %.10lf\n", avg_modpcoef);
    printf("Partition entropy: %.10lf (max: %.10lf)\n", avg_partent,
            log(clustc));
    printf("Average intra cluster distance: %.10lf\n", avg_aid);

    transpose_(&dists_t, &best_dists);
    print_header("Best instance indexes", HEADER_SIZE);
    printf("\nPartition coefficient: %.10lf\n", partcoef(&best_memb));
    printf("Modified partition coefficient: %.10lf\n",
            modpcoef(&best_memb));
    printf("Partition entropy: %.10lf (max: %.10lf)\n",
            partent(&best_memb), log(clustc));
    printf("Average intra cluster distance: %.10lf\n",
            avg_intra_dist(&best_memb, &dists_t, mfuz));

    print_header("Averaged crisp silhouette", HEADER_SIZE);
    print_silhouet(avg_csil);
    print_header("Averaged fuzzy silhouette", HEADER_SIZE);
    print_silhouet(avg_fsil);
    print_header("Averaged simple silhouette", HEADER_SIZE);
    print_silhouet(avg_ssil);

    aggregate_dmatrices(&agg_dmatrix, &best_weights);
    csil = crispsil(groups, &agg_dmatrix);
    print_header("Best instance crisp silhouette", HEADER_SIZE);
    print_silhouet(csil);
    fsil = fuzzysil(csil, groups, &best_memb, mfuz);
    print_header("Best instance fuzzy silhouette", HEADER_SIZE);
    print_silhouet(fsil);
    ssil = simplesil(pred, &dists_t);
    print_header("Best instance simple silhouette", HEADER_SIZE);
    print_silhouet(ssil);

    free_silhouet(avg_csil);
    free(avg_csil);
    free_silhouet(avg_fsil);
    free(avg_fsil);
    free_silhouet(avg_ssil);
    free(avg_ssil);
    free_silhouet(csil);
    free(csil);
    free_silhouet(fsil);
    free(fsil);
    free_silhouet(ssil);
    free(ssil);
    free(pred);
    free_st_matrix(groups);
    free(groups);
    free_st_matrix(&dists_t);
    free_st_matrix(&agg_dmatrix);
	for(i = 0; i < objc; ++i) {
		if(constraints[i]) {
            constraint_free(constraints[i]);
        }
	}
    free(constraints);
	for(i = 0; i < classc; ++i) {
		int_vec_free(&sample[i]);
	}
	free(sample);
END:
    fclose(stdout);
    for(j = 0; j < dmatrixc; ++j) {
        free_st_matrix(&dmatrix[j]);
    }
    free(dmatrix);
    free_st_matrix(&memb);
    free_st_matrix(&best_memb);
    for(k = 0; k < clustc; ++k) {
        free_st_matrix(&membvec[k]);
        free_st_matrix(&global_dmatrix[k]);
    }
    free(membvec);
    free(global_dmatrix);
    free_st_matrix(&dists);
    free_st_matrix(&best_dists);
    free_st_matrix(&weights);
    free_st_matrix(&best_weights);
	for(i = 0; i < classc; ++i) {
		int_vec_free(&class[i]);
	}
	free(class);
    free_st_matrix(&a_val);
//    free_st_matrix(&c_val);
//    free(c_val_obj);
    free(cluster_sum);
    free(sqd_memb_sum);
    return 0;
}
