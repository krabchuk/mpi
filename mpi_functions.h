#ifndef MPI_FUNCTIONS
#define MPI_FUNCTIONS

void main_block_reduce (double *main_block_norm, int *main_block_index, int p, int &main_block);

void mpi_matrix_mult (double *a, double *b, double *c, int n, int m, int p, int my_rank);

void mpi_matrix_read(const char *filename, double *a, int n, int m, int p, int my_rank, int max_columns, int &error);

void mpi_init_e (double *b, int n, int m, int p, int my_rank);

void mpi_matrix_print (double *a, int n, int m, int p, int my_rank, int max_columns);

void mpi_get_block (double *a, double *block, int i_local, int j_local, int n, int m, int p, int my_rank, int max_columns);

void mpi_add_massive (double *c, double *c_tmp, int i_local, int j_local, int n, int m, int p, int my_rank, int max_columns);

#endif // MPI_FUNCTIONS
