#ifndef HEAD_H
#define HEAD_H

#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_SIZE 10
#define EPS 1e-16

struct val_num
{
  double value;
  int numb;
};

// MPI HEADERS
/****************************/

int get_data(char *filename, double *a, double *b, int n, int m, int p, int my_rank);
void set_block (double *a, int global_i, int global_j, int n, int m);
int gauss_mpi (double *a, double *b, int n, int m, int my_rank, int p, int max_columns_global, double norm);
void set_diag_block (double *b, int n);

double matrix_norm_mpi (double *a, int n, int m, int p, int my_rank);  //stable

int search_main_block (double *a, int n, int m, int p, double norm_for_block, int my_rank, int i,
                   double *test_block, double *test_b, int *test_pos_for_block, double *main_block_norm);

double get_full_time();

/****************************/

void copy_massive (double *to, double *from, int size);
void add_massive (double *to, double *from, int size);

int read_matrix (double *a, int n, int m, FILE *fp);
void print_matrix (double *a, int n, int m);
void print_matrix_real (double *a, int n, int m);
void generate_matrix (double *a, int n, int m);
void init (double *a, int n, int m);
void print (double *a, int n, int m);

double neviazka (double *a, double *b, int n, int m);
void matrix_summ (double *a, double *b, int n, int m);

int gauss (double *a, double *b, int n, int *pos_j, double norm);
void swap_massive (double *a, double *b, int n, int m);
void swap_block (double *a, double *b, int n, int m);

int gauss_blocks (double *a, double *b, int n, int m, int *pos_j, double norm);
void matrix_multiply (double *a, double *b, double *c, int n, int m, int k);
void matrix_multiply_full (double *a, double *b, int n, int m);
void matrix_multiply_parted (double *a, double *b, double *c, int n, int m, int k);
void matrix_multiply_debug (double *a, double *b, int n, int m);


void copy_block (double *a, double *b, int n, int m);

void subtraction_from_matrix_a_matrix_b(double *a, double *b, int n, int m);
double block_norm (double *a, int n);

int pos_of_block (int i, int j, int n, int m);
void cpy_matrix (double *to, double *from, int n, int m);
void subtract_e (double *a, int n, int m);

double f1 (int i, int j);
double f2 (int i, int j);
double f3 (int i, int j);



#endif /* HEAD_H  */
