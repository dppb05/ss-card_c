// TODO:
//  - use only one object partition group from labels: exclude the
//  old one copied from SetMedoids-MLMNL-P and adapt the sampling
//  accordingly
//  - remove 'gen_sample_' and use 'gen_sample' from 'stex'
//  - compute mean 'CR' and 'F-measure' among instances
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

#include "util.h"
#include "matrix.h"
#include "stex.h"
#include "constraint.h"

#define BUFF_SIZE 1024
#define HEADER_SIZE 51

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
    size_t i;
    size_t k;
    st_matrix membvec_trans;
    init_st_matrix(&membvec_trans, 1, objc);
    st_matrix aux_mtx;
    init_st_matrix(&aux_mtx, 1, objc);
    st_matrix aux_mtx2;
    init_st_matrix(&aux_mtx2, 1, 1);
    st_matrix term1;
    init_st_matrix(&term1, objc, 1);
    double term2;
    bool hasneg = false;
    for(k = 0; k < clustc; ++k) {
        transpose_(&membvec_trans, &membvec[k]);
        mtxmult_(&aux_mtx, &membvec_trans, &global_dmatrix[k]);
        mtxmult_(&aux_mtx2, &aux_mtx, &membvec[k]);
        term2 = get(&aux_mtx2, 0, 0) * 0.5;
        mtxmult_(&term1, &global_dmatrix[k], &membvec[k]);
        for(i = 0; i < objc; ++i) {
            set(&dists, k, i, get(&term1, i, 0) - term2);
            if(!hasneg && get(&dists, k, i) < 0.0) {
                hasneg = true;
            }
        }
    }
    free_st_matrix(&membvec_trans);
    free_st_matrix(&aux_mtx);
    free_st_matrix(&aux_mtx2);
    free_st_matrix(&term1);
    return hasneg;
}

bool compute_dists_old() {
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
            val = sqdeuclid_dist(membvec[k].mtx, idcol, objc);
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

//void update_memb() {
//    size_t c;
//    size_t i;
//    size_t k;
//    double val;
//    int zerovalc;
//    printf("debug: update_memb start\n");
//    for(i = 0; i < objc; ++i) {
//        zerovalc = 0;
//        for(k = 0; k < clustc; ++k) {
//            if(deq_(get(&dists, k, i), 0.0)) {
//                ++zerovalc;
//            }
//        }
//        if(zerovalc) {
//            printf("Msg: there is at least one zero val for d[%d]."
//                    "\n", i);
//            val = 1.0 / ((double) zerovalc);
//            for(k = 0; k < clustc; ++k) {
//                if(get(&dists, k, i) > 0.0) {
//                    set(&memb, i, k, 0.0);
//                } else {
//                    set(&memb, i, k, val);
//                }
//            }
//        } else {
//            for(k = 0; k < clustc; ++k) {
//                val = 0.0;
//                for(c = 0; c < clustc; ++c) {
//                    val += pow(get(&dists, k, i) / get(&dists, c, i),
//                            1.0);
//                }
//                set(&memb, i, k, 1.0 / val);
//            }
//        }
//    }
//}

void constr_update_memb() {
    size_t c;
    size_t h;
    size_t i;
    size_t k;
    size_t obj;
    size_t zeroc;
    double val;
    double memb_sum;
    double memb_rfcm;
    double memb_constr;
    double c_val[clustc];
    double c_val_obj;
    double a_val_inv_sum;
    bool clip;
    for(i = 0; i < objc; ++i) {
        zeroc = 0;
        for(k = 0; k < clustc; ++k) {
            if(get(&a_val, i, k) <= 0.0) {
                ++zeroc;
            }
        }
        if(zeroc) {
//            val = 1.0 / (double) zeroc;
//            for(k = 0; k < clustc; ++k) {
//                if(get(&a_val, i, k) <= 0.0) {
//                    set(&memb, i, k, val);
//                } else {
//                    set(&memb, i, k, 0.0);
//                }
//            }
            continue;
        }
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
//        printf("Dispersions:\n");
        for(j = 0; j < dmatrixc; ++j) {
            val = 0.0;
            for(i = 0; i < objc; ++i) {
                for(h = 0; h < objc; ++h) {
                    val += pow(get(&memb, i, k), 2.0) *
                        pow(get(&memb, h, k), 2.0) *
                        get(&dmatrix[j], i, h);
                }
            }
            if(!val) {
                ++zeroc;
            }
            dispersion[j] = val;
//            printf("%.15lf ", val);
        }
//        printf("\n");
        if(zeroc) {
//            printf("Msg: at least one dispersion is zero for cluster"
//                    " %d\n", k);
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
    size_t j;
    size_t k;
    double sumw;
    double sum_num = 0.0;
    double sum_den = 0.0;
    for(i = 0; i < objc; ++i) {
        sum_den += pow(get(&memb, i, k), 2.0);
    }
    sum_den *= 2.0;
    for(k = 0; k < clustc; ++k) {
        for(i = 0; i < objc; ++i) {
            for(h = 0; h < objc; ++h) {
                sumw = 0.0;
                for(j = 0; j < dmatrixc; ++j) {
                    sumw += pow(get(&weights, k, j), qexp) *
                        get(&dmatrix[j], i, h);
                }
                sum_num += pow(get(&memb, i, k), 2.0) *
                    pow(get(&memb, h, k), 2.0) * sumw;
            }
        }
    }
    sum_num /= sum_den;
    sum_den = 0.0;
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
    obj = cmp(sum_den, 0.0);
//    printf("num: %.20lf\nden: %.20lf\ncmp: %d\n", sum_num, sum_den, obj);
    if(sum_den > FPOINT_OFFSET) {
//    if(dgt(sum_den, 0.0)) {
        alpha = sum_num / sum_den;
    } else {
        printf("Constraints fully satisfied, alpha unchanged.\n");
    }
}

double run() {
    printf("Initialization.\n");
    init_memb();
    if(verbose) print_memb(&memb);
    init_weights();
    if(verbose) print_weights(&weights);
//    compute_dists(); // pre-compute 'cluster_sum' and 'sqd_memb_sum'
    update_alpha();
    printf("Alpha: %.10f\n", alpha);
    beta = 0.0;
    double adeq = adequacy();
    printf("Adequacy: %.15lf\n", adeq);
    double prev_iter_adeq;
    double adeq_diff;
    size_t iter = 1;
    // st_matrix prev_memb;
    // init_st_matrix(&prev_memb, objc, clustc);
    do {
        printf("Iteration %d:\n", iter);
        prev_iter_adeq = adeq;
        global_dissim();
        compute_membvec();
        if(compute_dists()) {
            printf("Msg: applying b-spread transform\n");
            do {
                if(verbose) {
                    printf("Distances:\n");
                    print_st_matrix(&dists, 20, true);
                }
            } while(adjust_dists());
        }
        if(verbose) {
            printf("Distances:\n");
            print_st_matrix(&dists, 20, true);
        }
        adjust_a_val();
        update_alpha();
        printf("Alpha: %.10f\n", alpha);
        // mtxcpy(&prev_memb, &memb);
        constr_update_memb();
//        update_memb();
        if(verbose) print_memb(&memb);
        update_weights();
        if(verbose) print_weights(&weights);
        adeq = adequacy();
        printf("Adequacy: %.15lf\n", adeq);
        adeq_diff = prev_iter_adeq - adeq;
        if(adeq_diff < 0.0) {
//            adeq_diff = fabs(adeq_diff);
            printf("Warn: current iteration adequacy is greater "
                    "than previous(%.15lf).\n", -adeq_diff);
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
    // free_st_matrix(&prev_memb);
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

void gen_sample_(double sample_perc) {
	sample = malloc(sizeof(int_vec) * classc);
    size_t size;
	int pos;
    int val;
    size_t max;
	size_t i;
	size_t k;
	for(k = 0; k < classc; ++k) {
        max = class[k].size;
        size = max * sample_perc;
		int_vec_init(&sample[k], size);
		int objs[max];
        for(i = 0; i < max; ++i) {
            objs[i] = class[k].get[i];
        }
		for(i = 0; i < size; ++i) {
            pos = rand() % max;
            val = objs[pos];
            --max;
            objs[pos] = objs[max];
            objs[max] = val;
			int_vec_push(&sample[k], val);
		}
	}
}

void print_sample() {
	size_t i;
	size_t k;
	for(k = 0; k < classc; ++k) {
		printf("Class %d (%d members):\n", k, sample[k].size);
		for(i = 0; i < sample[k].size; ++i) {
			printf("%u ", sample[k].get[i]);
		}
		printf("\n");
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

bool dump_r_data(const char *filename, st_matrix *memb,
        st_matrix *weights, double alpha, double best_adeq) {
    FILE *outfile = fopen(filename, "w");
    if(!outfile) {
        return false;
    }
    size_t i;
    size_t j;
    size_t k;
    fprintf(outfile, "mvfcmv_model <- list(\n");
    // membership matrix start
    fprintf(outfile, "fuzzyMatrix = matrix(c(\n");
    size_t last = objc - 1;
    for(i = 0; i < last; ++i) {
        fprintf(outfile, "\t");
        for(k = 0; k < clustc; ++k) {
            fprintf(outfile, "%.10lf,", get(memb, i, k));
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\t");
    last = clustc - 1;
    for(k = 0; k < last; ++k) {
        fprintf(outfile, "%.10lf,", get(memb, i, k));
    }
    fprintf(outfile, "%.10lf\n),%d,%d,byrow=TRUE),\n", get(memb, i, k), objc, clustc);
    // weight matrix start
    fprintf(outfile, "weightMatrix = matrix(c(\n");
    for(k = 0; k < last; ++k) {
        fprintf(outfile, "\t");
        for(j = 0; j < dmatrixc; ++j) {
            fprintf(outfile, "%.10lf,", get(weights, k, j));
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\t");
    last = dmatrixc - 1;
    for(j = 0; j < last; ++j) {
        fprintf(outfile, "%.10lf,", get(weights, k, j));
    }
    fprintf(outfile, "%.10lf\n),%d,%d,byrow=TRUE),\n", get(weights, k, j), clustc,
            dmatrixc);
    // dist matrix start
    fprintf(outfile, "dists = matrix(c(\n");
    last = objc - 1;
    for(i = 0; i < last; ++i) {
        fprintf(outfile, "\t");
        for(k = 0; k < clustc; ++k) {
            fprintf(outfile, "%.10lf,", get(&dists, k, i));
        }
        fprintf(outfile, "\n");
    }
    fprintf(outfile, "\t");
    last = clustc - 1;
    for(k = 0; k < last; ++k) {
        fprintf(outfile, "%.10lf,", get(&dists, k, i));
    }
    fprintf(outfile, "%.10lf\n),%d,%d,byrow=TRUE),\n", get(&dists, k, i), objc, clustc);
    // adequacy start
    fprintf(outfile, "adequacy = %.10lf,", best_adeq);
    fprintf(outfile, "alpha = %.10lf)", alpha);
    fclose(outfile);
    return true;
}

int main(int argc, char **argv) {
    bool mean_idx = false;
    bool comp_idx = false;
    // If true, dumps the R data for each instance; otherwise only
    // the the best instance is dumped.
    bool dump_r_inst = true;
    verbose = false;
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
//    for(i = 0; i < objc; ++i) {
//        printf("%d ", labels[i]);
//    }
//    printf("\n");
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
    int seed;
    char seedstr[16];
    fscanf(cfgfile, "%s", seedstr);
    if(!strcmp(seedstr, "RAND")) {
        seed = time(NULL);
    } else {
        seed = atoi(seedstr);
    }
    char constr_str[BUFF_SIZE];
    fscanf(cfgfile, "%s", constr_str);
    double sample_perc = 0.0;
    char **constr_file_names = NULL;
    FILE *constr_file = fopen(constr_str, "r");
    if(!constr_file) {
      printf("Constraint does not seem to be a file, assuming float.\n");
      sample_perc = atof(constr_str);
      if(dlt(sample_perc, 0.0) || dgt(sample_perc, 1.0)) {
          printf("Error: sample_perc outside [0,1].\n");
          return 2;
      }
    } else {
      fclose(constr_file);
      constr_file_names = malloc(sizeof(char*) * insts);
      for(i = 0; i < insts; ++i) {
        if(i == 0) {
          constr_file_names[i] = constr_str;
        } else {
          constr_file_names[i] = malloc(sizeof(char) * BUFF_SIZE);
          fscanf(cfgfile, "%s", constr_file_names[i]);
        }
      }
    }
    fclose(cfgfile);
    freopen(outfilename, "w", stdout);
    printf("###Configuration summary:###\n");
    printf("Number of objects: %d\n", objc);
    printf("Number of clusters: %d\n", clustc);
    printf("Number of instances: %d\n", insts);
    printf("Maximum interations: %d\n", max_iter);
    printf("Parameter q: %lf\n", qexp);
    printf("Seed: %d\n", seed);
    printf("Sample perc: %lf\n", sample_perc);
    if(constr_file_names) {
      printf("Constraint files:\n");
      for(i = 0; i < insts; ++i) {
        printf("%s\n", constr_file_names[i]);
      }
    }
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
    st_matrix agg_dmatrix;
    if(comp_idx || mean_idx) {
        init_st_matrix(&dists_t, dists.ncol, dists.nrow);
        init_st_matrix(&agg_dmatrix, objc, objc);
    }
    silhouet *csil;
    silhouet *fsil;
    silhouet *ssil;
    silhouet *avg_csil;
    silhouet *avg_fsil;
    silhouet *avg_ssil;
    st_matrix *confmtx;
    int *pred;
    st_matrix *groups;
    srand(seed);
    size_t best_inst;
    double best_inst_adeq;
    double cur_inst_adeq;
  if(!constr_file) {
    gen_sample_(sample_perc);
    print_sample();
    constraints = gen_constraints(sample, classc, objc);
    print_constraints(constraints, objc);
  }
  char r_file_name[BUFF_SIZE];
    for(i = 1; i <= insts; ++i) {
      char *cntr_file_name = constr_file_names[i-1];
        if(constr_file_names) {
          printf("Opening %s\n", cntr_file_name);
          constr_file = fopen(cntr_file_name, "r");
          constraints = read_constr(constr_file, labels,
            classc, objc);
          print_constraints(constraints, objc);
          fclose(constr_file);
        }
        printf("Instance %d:\n", i);
        cur_inst_adeq = run();
        if(dump_r_inst) {
          char ext[BUFF_SIZE];
          sprintf(ext, "%02d.R", i);
          strcpy(r_file_name, outfilename); 
          replace_ext(r_file_name, ext);
          printf("Writing to %s\n", r_file_name);
          if(!dump_r_data(r_file_name, &memb, &weights, alpha,
            cur_inst_adeq)) {
            printf("Warn: Problem when dumping the data to the R file");
          }
        } else {
          print_weights(&weights);
          print_memb(&memb);
        }
        if(mean_idx) {
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
        }
        if(i == 1 || cur_inst_adeq < best_inst_adeq) {
            mtxcpy(&best_memb, &memb);
            mtxcpy(&best_dists, &dists);
            mtxcpy(&best_weights, &weights);
            best_inst_adeq = cur_inst_adeq;
            best_inst = i;
        }
        if(constr_file_names) {
          for(j = 0; j < objc; ++j) {
            if(constraints[j]) {
              constraint_free(constraints[j]);
            }
          }
          free(constraints);
          constraints = NULL;
        }
    }
	printf("\n");
    printf("Best adequacy %.15lf on instance %d.\n", best_inst_adeq,
            best_inst);
    printf("\n");
    print_memb(&best_memb);
    print_weights(&best_weights);
  // dumps the best config as an R script. Uses the output file name
  // as base
  if(!dump_r_inst) {
    replace_ext(outfilename, "R");
    dump_r_data(outfilename, &best_memb, &best_weights, alpha,
        best_inst_adeq);
  }
    if(comp_idx) {
        pred = defuz(&best_memb);
        // TODO: use 'asgroups' to make the constraints
        groups = asgroups(pred, objc, classc);
        print_header("Partitions", HEADER_SIZE);
        print_groups(groups);
        confmtx = confusion(labels, pred, objc);
        print_header("Best instance confusion matrix", HEADER_SIZE);
        print_st_matrix(confmtx, 0, true);
    }

    if(mean_idx) {
        print_header("Average indexes", HEADER_SIZE);
        printf("\nPartition coefficient: %.10lf\n", avg_partcoef);
        printf("Modified partition coefficient: %.10lf\n", avg_modpcoef);
        printf("Partition entropy: %.10lf (max: %.10lf)\n", avg_partent,
                log(clustc));
        printf("Average intra cluster distance: %.10lf\n", avg_aid);
    }

    if(comp_idx) {
        transpose_(&dists_t, &best_dists);
        print_header("Best instance indexes", HEADER_SIZE);
        printf("\nPartition coefficient: %.10lf\n", partcoef(&best_memb));
        printf("Modified partition coefficient: %.10lf\n",
                modpcoef(&best_memb));
        printf("Partition entropy: %.10lf (max: %.10lf)\n",
                partent(&best_memb), log(clustc));
        printf("Average intra cluster distance: %.10lf\n",
                avg_intra_dist(&best_memb, &dists_t, mfuz));
        printf("F-measure: %.10lf\n", fmeasure(confmtx, true));
        printf("CR: %.10lf\n", corand(labels, pred, objc));
        printf("NMI: %.10lf\n", nmi(confmtx));
    }

    if(mean_idx) {
        print_header("Averaged crisp silhouette", HEADER_SIZE);
        print_silhouet(avg_csil);
        print_header("Averaged fuzzy silhouette", HEADER_SIZE);
        print_silhouet(avg_fsil);
        print_header("Averaged simple silhouette", HEADER_SIZE);
        print_silhouet(avg_ssil);
    }

    if(comp_idx) {
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
    }

    if(mean_idx) {
        free_silhouet(avg_csil);
        free(avg_csil);
        free_silhouet(avg_fsil);
        free(avg_fsil);
        free_silhouet(avg_ssil);
        free(avg_ssil);
    }
    if(comp_idx || mean_idx) {
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
        free_st_matrix(confmtx);
        free(confmtx);
    }
  if(constraints) {
    for(i = 0; i < objc; ++i) {
      if(constraints[i]) {
        constraint_free(constraints[i]);
      }
    }
    free(constraints);
  }
  if(sample) {
    for(i = 0; i < classc; ++i) {
      int_vec_free(&sample[i]);
    }
  }
	free(sample);
END:
    fclose(stdout);
  if(constr_file_names) {
    // the first one does not need to be freed
    for(i = 1; i < insts; ++i) {
      free(constr_file_names[i]);
    }
    free(constr_file_names);
  }
    for(j = 0; j < dmatrixc; ++j) {
        free_st_matrix(&dmatrix[j]);
    }
    free(dmatrix);
    free_st_matrix(&memb);
    free_st_matrix(&best_memb);
    for(k = 0; k < clustc; ++k) {
        free_st_matrix(&membvec[k]);
    }
    free(membvec);
    free_st_matrix(&dists);
    free_st_matrix(&best_dists);
    free_st_matrix(&weights);
    free_st_matrix(&best_weights);
	for(i = 0; i < classc; ++i) {
		int_vec_free(&class[i]);
	}
	free(class);
    free_st_matrix(&a_val);
    free(cluster_sum);
    free(sqd_memb_sum);
    return 0;
}
