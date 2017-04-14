#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


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
mpi_matrix_mult (double *a, double *b, double *c, int n, int m, int p, int my_rank)
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
  double *a_tmp = new double [m * m];
  double *b_tmp = new double [m * m];

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
                      memset (a_tmp, 0, m * m * sizeof(double));
                      memset (b_tmp, 0, m * m * sizeof(double));
                      mpi_get_block (a, a_tmp, j, i, n, m, p, my_rank, max_columns);
                      mpi_get_block (b, b_tmp, i_global, column_b_in_proc, n, m, p, (my_rank + iter) % p, max_columns_b);
                      matrix_multiply (a_tmp, b_tmp, c_tmp, m, m, m);
                      //add_massive (c + column_b_in_proc * n * m + j * m * m, c_tmp, m * m);
                      mpi_add_massive (c, c_tmp, j, column_b_in_proc, n, m, p, (my_rank + iter) % p, max_columns_b);
                    }
                }

            }

          MPI_Sendrecv_replace (b, n * m * max_columns_global, MPI_DOUBLE, my_rank > 0 ? my_rank - 1 : p - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);
          MPI_Sendrecv_replace (c, n * m * max_columns_global, MPI_DOUBLE, my_rank == 0 ? p - 1 : my_rank - 1, 0,
                                my_rank == (p - 1) ? 0 : my_rank + 1, 0, MPI_COMM_WORLD, &status);

        }
    }

  delete [] a_tmp;
  delete [] b_tmp;
  delete [] c_tmp;
}

void
mpi_matrix_read (const char *filename, double *a, int n, int m, int p, int my_rank, int max_columns, int &error)
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
      if (my_rank == 0)
        printf ("Cant open file %s\n", filename);
      error = 1;
      return;
    }

  error = 0;

  if (my_rank == p - 1)
    {
      if (s != 0)
        {
          for (int i = 0; i < k; i++)
            {
              // Считываем по блочным строчкам
              memset (buf, 0, n * m * sizeof (double));

              if (error == 0)
                {
                  for (int pos_i = 0; pos_i < m; pos_i++)
                    {
                      for (int pos_j = 0; pos_j < n; pos_j++)
                        {
                          if (fscanf (fp, "%lf", buf + pos_i + pos_j * m) != 1)
                            {
                              printf ("Reading error in block %d %d\n", i, pos_j / m);
                              error = 1;
                              break;
                            }
                        }
                      if (error)
                        break;
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

              if (k % p == p - 1)
                {
                  // Себе не отправляем, а прост копируем
                  copy_massive (a + i * m * s + (k / p) * n * m, buf + k * m * m, m * s);
                }
              else
                {
                  MPI_Send (buf + k * m * m, m * s, MPI_DOUBLE, k % p, 0, MPI_COMM_WORLD);
                }

            }

          memset (buf, 0, n * m * sizeof (double));

          // Считываем последнюю строчку

          if (error == 0)
            {
              for (int pos_i = 0; pos_i < s; pos_i++)
                {
                  for (int pos_j = 0; pos_j < n; pos_j++)
                    {
                      if (fscanf (fp, "%lf", buf + pos_i + pos_j * s) != 1)
                        {
                          printf ("Reading error in block %d %d\n", k, pos_j / m);
                          error = 1;
                          break;
                        }
                    }
                  if (error)
                    break;
                }
            }


          for (int pos = 0; pos < k; pos++)
            {
              if (pos % p == p - 1)
                {
                  // Себе не отправляем, а прост копируем
                  copy_massive (a + k * m * m + (pos / p) * n * m, buf + pos * m * s, s * m);
                }
              else
                {
                  MPI_Send (buf + pos * s * m, s * m, MPI_DOUBLE, pos % p, 0, MPI_COMM_WORLD);
                }
            }

          if (k % p == p - 1)
            {
              // Себе не отправляем, а прост копируем
              copy_massive (a + k * m * s + (k / p) * n * m, buf + k * m * s, s * s);
            }
          else
            {
              MPI_Send (buf + k * m * s, s * s, MPI_DOUBLE, k % p, 0, MPI_COMM_WORLD);
            }

        }
      else  // s == 0
        {
          for (int i = 0; i < k; i++)
            {
              // Считываем по блочным строчкам
              memset (buf, 0, n * m * sizeof (double));

              if (error == 0)
                {
                  for (int pos_i = 0; pos_i < m; pos_i++)
                    {
                      for (int pos_j = 0; pos_j < n; pos_j++)
                        {
                          if (fscanf (fp, "%lf", buf + pos_i + pos_j * m) != 1)
                            {
                              printf ("Reading error in block %d %d\n", i, pos_j / m);
                              error = 1;
                              break;
                            }
                        }
                      if (error)
                        break;
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
  else    // my_rank != p - 1
    {
      if (s != 0)
        {
          for (int i = 0; i < k; i++)
            {
              if (my_rank == k % p)
                {
                  // Поток с крайним столбцом
                  // Цикл по строчкам
                  for (int columns = 0; columns < max_columns - 1; columns++)
                    {
                      // Цикл по столбцам (они не рядом, передавать будем по отдельности)
                      MPI_Recv (a + columns * n * m + i * m * m, m * m, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                    }
                  MPI_Recv (a + (max_columns - 1) * n * m + i * m * s, m * s, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                }
              else
                {
                  // Поток без крайнего столбца
                  // Цикл по строчкам
                  for (int columns = 0; columns < max_columns; columns++)
                    {
                      // Цикл по столбцам (они не рядом, передавать будем по отдельности)
                      MPI_Recv (a + columns * n * m + i * m * m, m * m, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                    }
                }

            }

          // Последняя строчка
          if (my_rank == k % p)
            {
              // Поток с крайним столбцом
              // Цикл по строчкам
              for (int columns = 0; columns < max_columns - 1; columns++)
                {
                  // Цикл по столбцам (они не рядом, передавать будем по отдельности)
                  MPI_Recv (a + columns * n * m + k * m * m, s * m, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                }
              MPI_Recv (a + (max_columns - 1) * n * m + k * m * s, s * s, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
            }
          else
            {
              // Поток без крайнего столбца
              // Цикл по строчкам
              for (int columns = 0; columns < max_columns; columns++)
                {
                  // Цикл по столбцам (они не рядом, передавать будем по отдельности)
                  MPI_Recv (a + columns * n * m + k * m * m, s * m, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                }
            }
        }
      else    // s == 0
        {
          for (int i = 0; i < k; i++)
            {
              // Цикл по строчкам
              for (int columns = 0; columns < max_columns; columns++)
                {
                  // Цикл по столбцам (они не рядом, передавать будем по отдельности)
                  MPI_Recv (a + columns * n * m + i * m * m, m * m, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
                }
            }

        }
    }

  delete [] buf;
}

void
mpi_init_e (double *b, int n, int m, int p, int my_rank)
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
          b += m * s;
        }
    }
  if (s != 0 && i == k)
    {
      for (j = 0; j < k; j++)
        {
          b += m * s;
        }
      set_diag_block (b, s);
    }
}


void
mpi_matrix_print (double *a, int n, int m, int p, int my_rank, int max_columns)
{

  if (n >= 10)
    return;

  int s = n % m;
  int k = n / m;

  double *full_matrix;

  if (my_rank == 0)
    {
      full_matrix = new double [n * n];
    }


  MPI_Status status;

  int max_columns_global = max_columns;

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

  if (s != 0)
    {
      if (my_rank == 0)
        {
          for (int i = 0; i < k; i++)
            {
              if (i % p == 0)
                {
                  copy_massive (full_matrix + i * n * m, a + (i / p) * m * n, m * n);
                }
              else
                {
                  MPI_Recv (full_matrix + i * n * m, n * m, MPI_DOUBLE, i % p, 0, MPI_COMM_WORLD, &status);
                }
            }
          if (k % p == 0)
            {
              copy_massive (full_matrix + k * n * m, a + (k / p) * m * n, n * s);
            }
          else
            {
              MPI_Recv (full_matrix + k * n * m, n * s, MPI_DOUBLE, k % p, 0, MPI_COMM_WORLD, &status);
            }
        }
      else
        {
          if (my_rank == k % p)
            {
              // Процесс с крайним столбцом
              for (int i = 0; i < max_columns - 1; i++)
                {
                  MPI_Send (a + i * m * n, m * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
              MPI_Send (a + (max_columns - 1) * m * n, s * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
          else
            {
              for (int i = 0; i < max_columns; i++)
                {
                  MPI_Send (a + i * m * n, m * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
    }
  else
    {
      if (my_rank == 0)
        {
          for (int i = 0; i < k; i++)
            {
              if (i % p == 0)
                {
                  copy_massive (full_matrix + i * n * m, a + (i / p) * m * n, m * n);
                }
              else
                {
                  MPI_Recv (full_matrix + i * n * m, n * m, MPI_DOUBLE, i % p, 0, MPI_COMM_WORLD, &status);
                }
            }
        }
      else
        {
          for (int i = 0; i < max_columns; i++)
            {
              MPI_Send (a + i * m * n, m * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

  if (my_rank == 0)
    {
      print_matrix (full_matrix, n, m);
      delete [] full_matrix;
    }



}


void
mpi_get_block (double *a, double *block, int i_local, int j_local, int n, int m, int p, int my_rank, int max_columns)
{
  int s = n % m;
  int k = n / m;

  memset (block, 0, m * m * sizeof (double));

  if (my_rank == k % p)
    {
      // крайний блок у этого процесса

      if (i_local == k)
        {
          if (j_local == (max_columns - 1))
            {
              for (int i = 0; i < s; i++)
                {
                  copy_massive (block + i * m, a + j_local * n * m + i_local * m * s + i * s, s);
                }
              for (int i = s; i < m; i++)
                {
                  block[i * m + i] = 1.;
                }
            }
          else
            {
              for (int i = 0; i < m; i++)
                {
                  copy_massive (block + i * m, a + j_local * n * m + i_local * m * m + i * s, s);
                }
            }
        }
      else   // i_local != k
        {

          if (j_local == (max_columns - 1))
            {
              copy_massive (block, a + j_local * n * m + i_local * m * s, m * s);
            }
          else
            {
              copy_massive (block, a + j_local * n * m + i_local * m * m, m * m);
            }
        }

    }
  else
    {
      if (i_local == k)
        {
          for (int i = 0; i < m; i++)
            {
              copy_massive (block + i * m, a + j_local * n * m + i_local * m * m + i * s, s);
            }
        }
      else
        {
          copy_massive (block, a + j_local * n * m + i_local * m * m, m * m);
        }
    }

}

void
mpi_add_massive (double *c, double *c_tmp, int i_local, int j_local, int n, int m, int p, int my_rank, int max_columns)
{
  int k = n / m;
  int s = n % m;



  if (my_rank == k % p)
    {
      // крайний блок у этого процесса

      if (i_local == k)
        {
          if (j_local == (max_columns - 1))
            {
              c = c + i_local * m * s + j_local * n * m;
              for (int j = 0; j < s; j++)
                {
                  for (int i = 0; i < s; i++)
                    {
                      *(c) = *(c) + *(c_tmp + i + j * m);
                      c = c + 1;
                    }
                }
            }
          else
            {
              c = c + i_local * m * m + j_local * n * m;
              for (int j = 0; j < m; j++)
                {
                  for (int i = 0; i < s; i++)
                    {
                      *(c) = *(c) + *(c_tmp + i + j * m);
                      c = c + 1;
                    }
                }
            }
        }
      else   // i_local != k
        {
          if (j_local == (max_columns - 1))
            {
              c = c + i_local * m * s + j_local * n * m;
              add_massive (c, c_tmp, m * s);
            }
          else
            {
              c = c + i_local * m * m + j_local * n * m;
              add_massive (c, c_tmp, m * m);
            }
        }

    }
  else
    {
      c = c + i_local * m * m + j_local * n * m;
      if (i_local == k)
        {
          for (int j = 0; j < m; j++)
            {
              for (int i = 0; i < s; i++)
                {
                  *(c) = *(c) + *(c_tmp + i + j * m);
                  c = c + 1;
                }
            }
        }
      else
        {
          add_massive (c, c_tmp, m * m);
        }
    }
}

void
mpi_subtract_e (double *a, int n, int m, int p, int my_rank, int max_columns_global)
{
  int k = n / m;
  int s = n % m;

  int max_columns = max_columns_global;

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

  if (my_rank == k % p && s != 0)
    {
      for (int column = 0; column < max_columns - 1; column++)
        {
          subtract_e_from_block (a + column * n * m + (column * p + my_rank) * m * m, m);
        }
      subtract_e_from_block (a + (max_columns - 1) * n * m + ((max_columns - 1) * p + my_rank) * m * s, s);
    }
  else
    {
      for (int column = 0; column < max_columns; column++)
        {
          subtract_e_from_block (a + column * n * m + (column * p + my_rank) * m * m, m);
        }
    }
}

void
subtract_e_from_block (double *a, int n)
{
  for (int i = 0; i < n; i++)
    a[i * n + i] -= 1.;
}

double
mpi_matrix_norm (double *a, int n, int m, int p, int my_rank, int max_columns_global)
{
  int k = n / m;
  int s = n % m;

  int max_columns = max_columns_global;

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

  double *summ_of_collumn = new double [m];

  double tmp_max = 0, max = 0;


  if (my_rank == k % p && s != 0)
    {
      for (int column = 0; column < max_columns - 1; column++)
        {
          memset (summ_of_collumn, 0, m * sizeof (double));
          for (int j = 0; j < m; j++)
            {
              for (int i = 0; i < m * k; i++)
                {
                  summ_of_collumn[j] += fabs (*(a + column * n * m + (i / m) * m * m + j * m + (i % m)));
                }
              for (int i = m * k; i < n; i++)
                {
                  summ_of_collumn[j] += fabs (*(a + column * n * m + (i / m) * m * m + j * s + (i % m)));
                }
            }
          tmp_max = summ_of_collumn[0];
          for (int j = 1; j < m; j++)
            if (tmp_max < summ_of_collumn[j])
              tmp_max = summ_of_collumn[j];
        }
      memset (summ_of_collumn, 0, m * sizeof (double));
      for (int j = 0; j < s; j++)
        {
          for (int i = 0; i < m * k; i++)
            {
              summ_of_collumn[j] += fabs (*(a + (max_columns - 1) * n * m + (i / m) * m * s + j * m + (i % m)));
            }
          for (int i = m * k; i < n; i++)
            {
              summ_of_collumn[j] += fabs (*(a + (max_columns - 1) * n * m + (i / m) * m * s + j * s + (i % m)));
            }
        }
      tmp_max = summ_of_collumn[0];
      for (int j = 1; j < s; j++)
        if (tmp_max < summ_of_collumn[j])
          tmp_max = summ_of_collumn[j];
    }
  else
    {
      for (int column = 0; column < max_columns; column++)
        {
          memset (summ_of_collumn, 0, m * sizeof (double));
          for (int j = 0; j < m; j++)
            {
              for (int i = 0; i < m * k; i++)
                {
                  summ_of_collumn[j] += fabs (*(a + column * n * m + (i / m) * m * m + j * m + (i % m)));
                }
              if (s != 0)
                {
                  for (int i = m * k; i < n; i++)
                    {
                      summ_of_collumn[j] += fabs (*(a + column * n * m + (i / m) * m * m + j * s + (i % m)));
                    }
                }
            }
          tmp_max = summ_of_collumn[0];
          for (int j = 1; j < m; j++)
            if (tmp_max < summ_of_collumn[j])
              {
                tmp_max = summ_of_collumn[j];
              }
        }
    }

  MPI_Allreduce (&tmp_max, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  delete [] summ_of_collumn;

  return max;
}
















































