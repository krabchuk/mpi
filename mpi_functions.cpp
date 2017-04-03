#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "head.h"
#include "mpi_functions.h"

void
main_block_reduce (double *main_block_norm, int *main_block_index, int p, int &main_block)
{
  int min = -1;
  double min_norm = -1;

  for (int i = 0; i < p; i++)
    {
      if ((min < 0 || min_norm > main_block_norm[i]) && main_block_index[i] >= 0)
        {
          min =  main_block_index[i];
          min_norm = main_block_norm[i];
        }
    }

  main_block = min;
}

void
matrix_mult_mpi (double *a, double *b, double *c, int n, int m, int p, int my_rank)
{
  int s = n % m;
  int k = n / m;
  int max_columns = 0;
  int max_columns_global = 0;
  int max_columns_b = 0;

  if (s != 0)
    {
      if ((k + 1) % p != 0)
        {
          max_columns_global = ((k + 1) / p) + 1;
        }
      else
        {
          max_columns_global = (k + 1) / p;
        }
    }
  else
    {
      if (k % p != 0)
        {
          max_columns_global = (k / p) + 1;
        }
      else
        {
          max_columns_global = k / p;
        }
    }

  double *c_tmp = new double [m * m];
  int i, j, iter;
  MPI_Status status;

  if (s != 0)
    {
      if (my_rank > k % p)
        {
          max_columns = max_columns_global - 1;
        }
      else
        {
          max_columns = max_columns_global;
        }
    }
  else
    {
      if (my_rank > (k - 1) % p)
        {
          max_columns = max_columns_global - 1;
        }
      else
        {
          max_columns = max_columns_global;
        }
    }

#if 0
  if (s != 0)
    {
      for (iter = 0; iter < p; iter++)
        {
          // Сколько колонок у процесса сейчас
          if (s != 0)
            {
              if (((my_rank + iter) % p) > k % p)
                {
                  max_columns_b = max_columns_global - 1;
                }
              else
                {
                  max_columns_b = max_columns_global;
                }
            }
          else
            {
              if (((my_rank + iter) % p) > (k - 1) % p)
                {
                  max_columns_b = max_columns_global - 1;
                }
              else
                {
                  max_columns_b = max_columns_global;
                }
            }

          if ((my_rank + iter) % p == k % p)
            {
              // Процесс с крайним столбцом
              for (i = 0; i < max_columns - 1; i++)
                {
                  int i_global = i * p + my_rank;
                  for (int column_b_in_proc = 0; column_b_in_proc < max_columns_b - 1; column_b_in_proc++)
                    {
                      for (j = 0; j < k; j++)
                        {
                          matrix_multiply (a + i * n * m + j * m * m, b + column_b_in_proc * n * m + i_global * m * m,
                                           c_tmp, m, m, m);
                          add_massive (c + column_b_in_proc * n * m + j * m * m, c_tmp, m * m);
                        }

                      matrix_multiply (a + i * n * m + k * m * m, b + column_b_in_proc * n * m + i_global * m * m,
                                       c_tmp, s, m, m);
                      add_massive (c + column_b_in_proc * n * m + k * m * m, c_tmp, s * m);

                    }
                  // column_b_in_proc = max_columns - 1
                  for (j = 0; j < k; j++)
                    {
                      matrix_multiply (a + i * n * m + j * m * m, b + (max_columns_b - 1) * n * m + i_global * m * s,
                                       c_tmp, m, m, s);
                      add_massive (c + (max_columns_b - 1) * n * m + j * m * s, c_tmp, m * s);
                    }

                  matrix_multiply (a + i * n * m + k * m * m, b + (max_columns_b - 1) * n * m + i_global * m * s,
                                   c_tmp, s, m, s);
                  add_massive (c + (max_columns_b - 1) * n * m + k * m * s, c_tmp, s * s);

                }

              // i = max_columns - 1
              int i_global = (max_columns - 1) * p + my_rank;                          // < вроде как это должно получится k

              for (int column_b_in_proc = 0; column_b_in_proc < max_columns - 1; column_b_in_proc++)
                {
                  for (j = 0; j < k; j++)
                    {
                      matrix_multiply (a + (max_columns - 1) * n * m + j * m * s, b + column_b_in_proc * n * m + i_global * m * m,
                                       c_tmp, m, s, m);
                      add_massive (c + column_b_in_proc * n * m + j * m * m, c_tmp, m * m);
                    }

                  matrix_multiply (a + (max_columns - 1) * n * m + k * m * s, b + column_b_in_proc * n * m + i_global * m * m,
                                   c_tmp, s, s, m);
                  add_massive (c + column_b_in_proc * n * m + k * m * m, c_tmp, s * m);

                }
              // column_b_in_proc = max_columns - 1
              for (j = 0; j < k; j++)
                {
                  matrix_multiply (a + (max_columns - 1) * n * m + j * m * s, b + (max_columns - 1) * n * m + i_global * m * s,
                                   c_tmp, m, s, s);
                  add_massive (c + (max_columns - 1) * n * m + j * m * s, c_tmp, m * s);
                }

              matrix_multiply (a + (max_columns - 1) * n * m + k * m * s, b + (max_columns - 1) * n * m + i_global * m * s,
                               c_tmp, s, s, s);
              add_massive (c + (max_columns - 1) * n * m + k * m * s, c_tmp, s * s);

            }
          else
            {
              for (i = 0; i < max_columns; i++)
                {
                  int i_global = i * p + my_rank;
                  for (int column_b_in_proc = 0; column_b_in_proc < max_columns; column_b_in_proc++)
                    {
                      for (j = 0; j < k; j++)
                        {
                          printf ("Mult matrices:\n");
                          print (a + i * n * m + j * m * m, m, m);
                          print (b + column_b_in_proc * n * m + i_global * m * m, m, m);

                          matrix_multiply (a + i * n * m + j * m * m, b + column_b_in_proc * n * m + i_global * m * m,
                                           c_tmp, m, m, m);
                          add_massive (c + column_b_in_proc * n * m + j * m * m, c_tmp, m * m);
                        }

                      matrix_multiply (a + i * n * m + k * m * m, b + column_b_in_proc * n * m + i_global * m * m,
                                       c_tmp, s, m, m);
                      add_massive (c + column_b_in_proc * n * m + k * m * m, c_tmp, s * m);

                    }

                }
            }

          MPI_Sendrecv_replace (b, n * m * max_columns_global, MPI_DOUBLE, my_rank > 0 ? my_rank - 1 : p - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);
          MPI_Sendrecv_replace (c, n * m * max_columns_global, MPI_DOUBLE, my_rank == 0 ? p - 1 : my_rank - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);

        }
    }
  else
    {
      for (iter = 0; iter < p; iter++)
        {

              for (i = 0; i < max_columns; i++)
                {
                  int i_global = i * p + my_rank;
                  for (int column_b_in_proc = 0; column_b_in_proc < max_columns; column_b_in_proc++)
                    {
                      for (j = 0; j < k; j++)
                        {
                          matrix_multiply (a + i * n * m + j * m * m, b + column_b_in_proc * n * m + i_global * m * m,
                                           c_tmp, m, m, m);
                          add_massive (c + column_b_in_proc * n * m + j * m * m, c_tmp, m * m);
                        }
                    }

                }

          MPI_Sendrecv_replace (b, n * m * max_columns_global, MPI_DOUBLE, my_rank > 0 ? my_rank - 1 : p - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);
          MPI_Sendrecv_replace (c, n * m * max_columns_global, MPI_DOUBLE, my_rank == 0 ? p - 1 : my_rank - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);

        }
    }
#endif

  if (s == 0)
    {
      for (iter = 0; iter < p; iter++)
        {
          // Сколько колонок у процесса сейчас
          if (s != 0)
            {
              if (((my_rank + iter) % p) > k % p)
                {
                  max_columns_b = max_columns_global - 1;
                }
              else
                {
                  max_columns_b = max_columns_global;
                }
            }
          else
            {
              if (((my_rank + iter) % p) > (k - 1) % p)
                {
                  max_columns_b = max_columns_global - 1;
                }
              else
                {
                  max_columns_b = max_columns_global;
                }
            }


          for (i = 0; i < max_columns; i++)
            {
              int i_global = i * p + my_rank;
              for (int column_b_in_proc = 0; column_b_in_proc < max_columns_b; column_b_in_proc++)
                {
                  for (j = 0; j < k; j++)
                    {
                      matrix_multiply (a + i * n * m + j * m * m, b + column_b_in_proc * n * m + i_global * m * m,
                                       c_tmp, m, m, m);
                      add_massive (c + column_b_in_proc * n * m + j * m * m, c_tmp, m * m);
                    }
                }

            }

          MPI_Sendrecv_replace (b, n * m * max_columns_global, MPI_DOUBLE, my_rank > 0 ? my_rank - 1 : p - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);
          MPI_Sendrecv_replace (c, n * m * max_columns_global, MPI_DOUBLE, my_rank == 0 ? p - 1 : my_rank - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);

        }
    }
  else
    {
      n = (k + 1) * m;
      // дополнительняя крайняя колонка


      for (iter = 0; iter < p; iter++)
        {


          // Сколько колонок у процесса сейчас
          if (s != 0)
            {
              if (((my_rank + iter) % p) > k % p)
                {
                  max_columns_b = max_columns_global - 1;
                }
              else
                {
                  max_columns_b = max_columns_global;
                }
            }
          else
            {
              if (((my_rank + iter) % p) > (k - 1) % p)
                {
                  max_columns_b = max_columns_global - 1;
                }
              else
                {
                  max_columns_b = max_columns_global;
                }
            }



          for (i = 0; i < max_columns; i++)
            {
              int i_global = i * p + my_rank;
              for (int column_b_in_proc = 0; column_b_in_proc < max_columns_b; column_b_in_proc++)
                {
                  for (j = 0; j < k + 1; j++)
                    {
                      matrix_multiply (a + i * n * m + j * m * m, b + column_b_in_proc * n * m + i_global * m * m,
                                       c_tmp, m, m, m);
                      //if (my_rank == 0) print (c_tmp, m, m);
                      add_massive (c + column_b_in_proc * n * m + j * m * m, c_tmp, m * m);
                    }
                }

            }

          MPI_Sendrecv_replace (b, n * m * max_columns_global, MPI_DOUBLE, my_rank > 0 ? my_rank - 1 : p - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);
          MPI_Sendrecv_replace (c, n * m * max_columns_global, MPI_DOUBLE, my_rank == 0 ? p - 1 : my_rank - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);

        }
    }

  delete c_tmp;
}

int
matrix_read_mpi (const char *filename, double *a, int n, int m, int p, int my_rank, int max_columns)
{
  FILE *fp = fopen (filename, "r");

  MPI_Status status;

  int k = n / m;
  int s = n % m;

  int max_columns_global = max_columns;

  double *buf = new double [n * m];

  if (s != 0)
    {
      if (my_rank > k % p)
        {
          max_columns = max_columns_global - 1;
        }
      else
        {
          max_columns = max_columns_global;
        }
    }
  else
    {
      if (my_rank > (k - 1) % p)
        {
          max_columns = max_columns_global - 1;
        }
      else
        {
          max_columns = max_columns_global;
        }
    }

  if (!fp)
    {
      printf ("Cant open file %s\n", filename);
      return -1;
    }

  if (my_rank == p - 1)
    {
      if (s != 0)
        {


          for (int i = 0; i < k; i++)
            {
              // Считываем по блочным строчкам
              memset (buf, 0, n * m * sizeof (double));

              for (int pos_i = 0; pos_i < m; pos_i++)
                {
                  for (int pos_j = 0; pos_j < n; pos_j++)
                    {
                      if (fscanf (fp, "%lf", buf + pos_i + pos_j * m) != 1)
                        {
                          printf ("Reading error %d %d\n", pos_i, pos_j);
                        }
                    }
                }

              for (int pos = 0; pos < k + 1; pos++)
                {
                  if (pos % p == p - 1)
                    {
                      // Себе не отправляем, а прост копируем
                      copy_massive (a + i * m * m + (pos / p) * (k + 1) * m * m, buf + pos * m * m, m * m);
                    }
                  else
                    {
                      MPI_Send (buf + pos * m * m, m * m, MPI_DOUBLE, pos % p, 0, MPI_COMM_WORLD);
                    }
                }
            }

          memset (buf, 0, n * m * sizeof (double));

          // Считываем последнюю строчку
          for (int pos_i = 0; pos_i < s; pos_i++)
            {
              for (int pos_j = 0; pos_j < n; pos_j++)
                {
                  if (fscanf (fp, "%lf", buf + pos_i + pos_j * m) != 1)
                    {
                      printf ("Reading error %d %d\n", pos_i, pos_j);
                    }
                }
            }


          for (int pos = 0; pos < k + 1; pos++)
            {
              if (pos % p == p - 1)
                {
                  // Себе не отправляем, а прост копируем
                  copy_massive (a + k * m * m + (pos / p) * (k + 1) * m * m, buf + pos * m * m, m * m);
                }
              else
                {
                  MPI_Send (buf + pos * m * m, m * m, MPI_DOUBLE, pos % p, 0, MPI_COMM_WORLD);
                }
            }

        }
      else
        {
          for (int i = 0; i < k; i++)
            {
              // Считываем по блочным строчкам

              for (int pos_i = 0; pos_i < m; pos_i++)
                {
                  for (int pos_j = 0; pos_j < n; pos_j++)
                    {
                      fscanf (fp, "%lf", buf + pos_i + pos_j * m);
                    }
                }

              for (int pos = 0; pos < k; pos++)
                {
                  if (pos % p == p - 1)
                    {
                      // Себе не отправляем, а прост копируем
                      copy_massive (a + i * m * m + (pos / p) * n * m, buf + pos * m * m, m * m);
                    }
                  else
                    {
                      MPI_Send (buf + pos * m * m, m * m, MPI_DOUBLE, pos % p, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }


    }
  else
    {
      if (s != 0)
        {
          for (int i = 0; i < k + 1; i++)
            {
              // Цикл по строчкам
              for (int columns = 0; columns < max_columns; columns++)
                {
                  // Цикл по столбцам (они не рядом, передавать будем по отдельности)
                  MPI_Recv (a + (k + 1) * m * m * columns + m * m * i, m * m, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                }
            }
        }
      else
        {
          for (int i = 0; i < k; i++)
            {
              // Цикл по строчкам
              for (int columns = 0; columns < max_columns; columns++)
                {
                  // Цикл по столбцам (они не рядом, передавать будем по отдельности)
                  MPI_Recv (a + n * m * columns + m * m * i, m * m, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                }
            }

        }
    }

  if (s != 0)
    {
      if (my_rank == k % p)
        {
          for (int i = s; i < m; i++)
            {
              *(a + k * m * m + (max_columns - 1) * m * m * (k + 1) + i * m + i) = 1.;
            }
        }
    }

  delete buf;

  return 0;
}

void
init_e_mpi (double *b, int n, int m, int p, int my_rank)
{
  int i, j, k, s;

  k = n / m;
  s = n % m;


  for (i = my_rank; i < k; i += p)
    {
      for (j = 0; j < k; j++)
        {
          if (i == j)
            {
              set_diag_block (b, m);
            }
          b += m * m;
        }
      if (s)
        {
          b += m * m;
        }
    }
  if (s != 0 && i == k)
    {
      for (j = 0; j < k; j++)
        {
          b += m * m;
        }
      set_diag_block (b, m);
    }
}

























































