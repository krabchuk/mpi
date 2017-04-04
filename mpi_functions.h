#ifndef MPI_FUNCTIONS
#define MPI_FUNCTIONS

void main_block_reduce (double *main_block_norm, int *main_block_index, int p, int &main_block);

void matrix_mult_mpi (double *a, double *b, double *c, int n, int m, int p, int my_rank);

void matrix_read_mpi(const char *filename, double *a, int n, int m, int p, int my_rank, int max_columns, int &error);

void init_e_mpi (double *b, int n, int m, int p, int my_rank);

void matrix_print_mpi (double *a, int n, int m, int p, int my_rank, int max_columns);

#endif // MPI_FUNCTIONS
