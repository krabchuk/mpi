#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>


#include "mpi_functions.h"
#include "head.h"

void
copy_massive (double *to, double *from, int size)
{
  for (int i = 0; i < size; i++)
    {
      to[i] = from[i];
    }
}

void
add_massive (double *to, double *from, int size)
{
  for (int i = 0; i < size; i++)
    {
      to[i] += from[i];
    }
}

int
pos_of_block (int i, int j, int n, int m)
{
  int s, k;
  k = n / m;
  s = n % m;
  if (j == k)
    return j * m * n + i * m * s;
  else
    return j * m * n + i * m * m;
}

void
cpy_matrix (double *to, double *from, int n, int m)
{
  int i, j;
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          to[i + j * n] = from[i + j * n];
        }
    }
}

void
matrix_multiply_full (double *a, double *b, int n, int m)
{
  /* Multiplying of matriсes, a * b, result stay in a */

  int i, j, l, s, k, sum_i, sum_j;
  double *c, *sum, *buf;

  k = n / m;
  s = n % m;

  c = new double [n * n];
  sum = new double [m * m];
  buf = new double [m * m];

  for (i = 0; i < k; i++)
    {
      for (j = 0; j < k; j++)
        {
          for (l = 0; l < m * m; l++)
            {
              sum[l] = 0.0;
            }

          for (l = 0; l < k; l++)
            {
              matrix_multiply (a + pos_of_block (i, l, n, m), b + pos_of_block (l, j, n, m), buf, m, m, m);
              for (sum_i = 0; sum_i < m; sum_i++)
              {
                for (sum_j = 0; sum_j < m; sum_j++)
                  {
                    sum[sum_i + sum_j * m] += buf[sum_i + sum_j * m];
                  }
              }
            }
          if (s != 0)
            {
              matrix_multiply (a + pos_of_block (i, k, n, m), b + pos_of_block (k, j, n, m), buf, m, s, m);
              for (sum_i = 0; sum_i < m; sum_i++)
                {
                  for (sum_j = 0; sum_j < m; sum_j++)
                    {
                      sum[sum_i + sum_j * m] += buf[sum_i + sum_j * m];
                    }
                }
            }

          cpy_matrix (c + pos_of_block (i, j, n, m), sum, m, m);
        }
    }

  for (i = 0; i < k; i++)
    {
      for (l = 0; l < m * s; l++)
        {
          sum[l] = 0.0;
        }

      for (l = 0; l < k; l++)
        {
          matrix_multiply (a + pos_of_block (i, l, n, m), b + pos_of_block (l, k, n, m), buf, m, m, s);
          for (sum_i = 0; sum_i < m; sum_i++)
            {
              for (sum_j = 0; sum_j < s; sum_j++)
                {
                  sum[sum_i + sum_j * m] += buf[sum_i + sum_j * m];
                }
            }
        }
      matrix_multiply (a + pos_of_block (i, k, n, m), b + pos_of_block (k, k, n, m), buf, m, s, s);
      for (sum_i = 0; sum_i < m; sum_i++)
        {
          for (sum_j = 0; sum_j < s; sum_j++)
            {
              sum[sum_i + sum_j * m] += buf[sum_i + sum_j * m];
            }
        }

      cpy_matrix (c + pos_of_block (i, k, n, m), sum, m, s);
    }


  for (j = 0; j < k; j++)
    {
      for (l = 0; l < m * m; l++)
        {
          sum[l] = 0.0;
        }

      for (l = 0; l < k; l++)
        {
          matrix_multiply (a + pos_of_block (k, l, n, m), b + pos_of_block (l, j, n, m), buf, s, m, m);
          for (sum_i = 0; sum_i < s; sum_i++)
            {
              for (sum_j = 0; sum_j < m; sum_j++)
                {
                  sum[sum_i + sum_j * s] += buf[sum_i + sum_j * s];
                }
            }
        }
      matrix_multiply (a + pos_of_block (k, k, n, m), b + pos_of_block (k, j, n, m), buf, s, s, m);
      for (sum_i = 0; sum_i < s; sum_i++)
        {
          for (sum_j = 0; sum_j < m; sum_j++)
            {
              sum[sum_i + sum_j * s] += buf[sum_i + sum_j * s];
            }
        }


      cpy_matrix (c + pos_of_block (k, j, n, m), sum, s, m);
    }

  for (l = 0; l < m * m; l++)
    {
      sum[l] = 0.0;
    }

  for (i = 0; i < k; i++)
    {
      matrix_multiply (a + pos_of_block (k, i, n, m), b + pos_of_block (i, k, n, m), buf, s, m, s);
      for (sum_i = 0; sum_i < s; sum_i++)
        {
          for (sum_j = 0; sum_j < s; sum_j++)
            {
              sum[sum_i + sum_j * s] += buf[sum_i + sum_j * s];
            }
        }
    }
  matrix_multiply (a + pos_of_block (k, k, n, m), b + pos_of_block (k, k, n, m), buf, s, s, s);
  for (sum_i = 0; sum_i < s; sum_i++)
    {
      for (sum_j = 0; sum_j < s; sum_j++)
        {
          sum[sum_i + sum_j * s] += buf[sum_i + sum_j * s];
        }
    }
  cpy_matrix (c + pos_of_block (k, k, n, m), sum, s, s);


  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        {
          a[i + j * n] = c[i + j * n];
        }
    }

  delete [] sum;
  delete [] buf;
  delete [] c;
}

void
matrix_multiply_parted (double *a, double *b, double *c, int n, int m, int k)
{
  /* Multiplying of matriсes, a * b, result stay in a
   * a = n * m, b = m * k */
  int i, j, l;
  double s00;

  if (n == 0 || m == 0 || k == 0)
    {
      return;
    }

  for (i = 0; i < n; i ++)
    {
      for (j = 0; j < k; j ++)
        {
          s00 = 0.0;
          for (l = 0; l < m; l++)
            {
              s00 += a[i + l * n] * b[l + j * m];
            }
          c[i + j * n] = s00;
        }
    }
}

void
matrix_multiply (double *a, double *b, double *c, int n, int m, int k)
{
  /* Multiplying of matriсes, a * b
   * a = n * m, b = m * k */
  int i, j, l, s_i, s_j;
  double s00, s01, s02, s10, s11, s12, s20, s21, s22;

  s_i = n % 3;
  s_j = k % 3;

  if (n == 0 || m == 0 || k == 0)
    {
      return;
    }


  if (n > 2 && k > 2)
    {
      for (i = 0; i + 2 < n; i += 3)
        {
          for (j = 0; j + 2 < k; j += 3)
            {

              s00 = 0.0;
              s01 = 0.0;
              s02 = 0.0;
              s10 = 0.0;
              s11 = 0.0;
              s12 = 0.0;
              s20 = 0.0;
              s21 = 0.0;
              s22 = 0.0;

              for (l = 0; l < m; l++)
                {
                  s00 += a[i + l * n] * b[l + j * m];
                  s01 += a[i + l * n] * b[l + (1 + j) * m];
                  s02 += a[i + l * n] * b[l + (2 + j) * m];
                  s10 += a[i + 1 + l * n] * b[l + j * m];
                  s11 += a[i + 1 + l * n] * b[l + (1 + j) * m];
                  s12 += a[i + 1 + l * n] * b[l + (2 + j) * m];
                  s20 += a[i + 2 + l * n] * b[l + j * m];
                  s21 += a[i + 2 + l * n] * b[l + (1 + j) * m];
                  s22 += a[i + 2 + l * n] * b[l + (2 + j) * m];
                }

              c[i + j * n] = s00;
              c[i + (j + 1) * n] = s01;
              c[i + (j + 2) * n] = s02;
              c[i + 1 + j * n] = s10;
              c[i + 1 + (j + 1) * n] = s11;
              c[i + 1 + (j + 2) * n] = s12;
              c[i + 2 + j * n] = s20;
              c[i + 2 + (j + 1) * n] = s21;
              c[i + 2 + (j + 2) * n] = s22;
            }
        }
    }
  for (i = n - s_i; i < n; i++)
    {
      for (j = 0; j < k - s_j; j++)
        {
          s00 = 0.0;
          for (l = 0; l < m; l++)
            {
              s00 += a[i + l * n] * b[l + j * m];
            }
          c[i + j * n] = s00;
        }
    }

  for (j = k - s_j; j < k; j++)
    {
      for (i = 0; i < n - s_i; i++)
        {
          s00 = 0.0;
          for (l = 0; l < m; l++)
            {
              s00 += a[i + l * n] * b[l + j * m];
            }
          c[i + j * n] = s00;
        }
    }

  for (i = n - s_i; i < n; i++)
    {
      for (j = k - s_j; j < k; j++)
        {
          s00 = 0.0;
          for (l = 0; l < m; l++)
            {
              s00 += a[i + l * n] * b[l + j * m];
            }
          c[i + j * n] = s00;
        }
    }
}

double
matrix_norm_mpi (double *a,
             int n,
             int m,
             int p,
             int my_rank)
{
  double *summ_of_collumn;
  double max = -1;
  double try_max;
  int k = 0;
  int s = 0;

  k = n / m;
  s = n % m;

  summ_of_collumn = new double [m];
  memset (summ_of_collumn, 0, m * sizeof (double));

  //columns
  for (int j = my_rank; j < k; j += p)
    {
      //rows
      for (int i = 0; i < k; i++)
        {
          //summ in block
          for (int j_block = 0; j_block < m; j_block++)
            {
              for (int i_block = 0; i_block < m; i_block++)
                {
                  summ_of_collumn[j_block] += fabs (*a);
                  a++;
                }
            }
        }
      if (s != 0)
        {
          //summ in last block
          for (int j_block = 0; j_block < m; j_block++)
            {
              for (int i_block = 0; i_block < s; i_block++)
                {
                  summ_of_collumn[j_block] += fabs (*a);
                  a++;
                }
            }
        }
      //search max
      for (int i = 0; i < m; i++)
        {
          try_max = summ_of_collumn[i];
          if (max < try_max)
            {
              max = try_max;
            }
        }
      memset (summ_of_collumn, 0, m * sizeof (double));
    }
  //last small collumn
  if (s != 0)
    {
      //rows
      for (int i = 0; i < k; i++)
        {
          //summ in block
          for (int j_block = 0; j_block < s; j_block++)
            {
              for (int i_block = 0; i_block < m; i_block++)
                {
                  summ_of_collumn[j_block] += fabs (*a);
                  a++;
                }
            }
        }
      //summ in last block
      for (int j_block = 0; j_block < s; j_block++)
        {
          for (int i_block = 0; i_block < s; i_block++)
            {
              summ_of_collumn[j_block] += fabs (*a);
              a++;
            }
        }
      //search max
      for (int i = 0; i < s; i++)
        {
          try_max = summ_of_collumn[i];
          if (max < try_max)
            {
              max = try_max;
            }
        }
    }

  delete [] summ_of_collumn;

  MPI_Allreduce (&max, &try_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return try_max;
}


double
block_norm (double *a, int n)
{
  int i, j;
  double summ, max = -1;

  for (i = 0; i < n; i++)
    {
      summ = 0;
      for (j = 0; j < n; j++)
        {
          summ += fabs (a[i + j * n]);
        }
      if (summ > max)
        max = summ;
    }

  return max;
}

void
subtraction_from_matrix_a_matrix_b (double *a, double *b, int n, int m)
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          a[i + j * n] -= b[i + j * n];
        }
    }
}

int
gauss (double *a, double *b, int n, int *pos_j, double norm)
{
  int i, j, k, swap, indMax;
  double tmp, max;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        {
          b[i + j * n] = (double)(i == j);
        }
      pos_j[i] = i;
    }
  for (i = 0; i < n; i++)
  {
      max = fabs(a[i + pos_j[i] * n]);
      indMax = i;
      for (j = i + 1; j < n; ++j)
        {
          if (max < fabs(a[i + pos_j[j] * n]))
            {
              max = fabs(a[i + pos_j[j] * n]);
              indMax = j;
            }
        }

      swap = pos_j[i];
      pos_j[i] = pos_j[indMax];
      pos_j[indMax] = swap;
      if (fabs(a[i + pos_j[i] * n]) < norm * 1e-15)
        {
          return 0;
        }

      tmp = 1.0 / a[i + pos_j[i] * n];

      for (j = i; j < n; j++)
        {
          a[i + pos_j[j] * n] *= tmp;
        }

      for (j = 0; j < n; j++)
        {
          b[i + j * n] *= tmp;
        }

      for (j = i + 1; j < n; j++)
        {
          tmp = a[j + pos_j[i] * n];

          for (k = i; k < n; k++)
            {
              a[j + pos_j[k] * n] -= a[i + pos_j[k] * n] * tmp;
            }

          for (k = 0; k < n; k++)
            {
              b[j + k * n] -= b[i + k * n] * tmp;
            }
        }
  }

  for (k = 0; k < n; k++)
    {
      for (i = n - 1; i >= 0; --i)
        {
          tmp = b[i + k * n];
          for (j = i + 1; j < n; ++j)
            {
              tmp -= a[i + pos_j[j] * n] * b[j + k * n];
            }
          b[i + k * n] = tmp;
        }
    }

  for (i = 0; i < n; ++i)
    {
      for (j = 0; j < n; ++j)
        {
          a[pos_j[i] + j * n] = b[i + j * n];
        }
    }

  for (i = 0; i < n; ++i)
    {
      for (j = 0; j < n; ++j)
        {
          b[i + j * n] = a[i + j * n];
        }
    }
  return 1;
}


int
search_main_block (double *a,
                   int n,
                   int m,
                   int p,
                   double norm_for_block,
                   int my_rank,
                   int i,
                   double *test_block,
                   double *test_b,
                   int *test_pos_for_block,
                   // return value
                   double *main_block_norm)
{
  int start, min = -1, k;
  double min_norm = -1, norm = -1;

  k = n / m;

  if (my_rank >= i % p)
    {
      start = i / p;
    }
  else
    {
      start = i / p + 1;
    }

  a =  a + start * m * n + i * m * m;

  if (my_rank < i % p)
    {
      start = i - i % p + p + my_rank;
    }
  else
    {
      if (my_rank > i % p)
        {
          start = i - i % p + my_rank;
        }
      else
        {
          start = i;
        }
    }




  int iter = start;


  for (iter = start; iter < k; iter += p)
    {
      cpy_matrix (test_block, a, m, m);
      if (gauss (test_block, test_b, m, test_pos_for_block, norm_for_block))
        {
          min = iter;
          min_norm = block_norm (test_b, m);
          iter += p;
          a = a + n * m;
          break;
        }
      a = a + n * m;
    }

  for (; iter < k; iter += p)
    {
      cpy_matrix (test_block, a, m, m);
      if (gauss (test_block, test_b, m, test_pos_for_block, norm_for_block))
        {
          norm = block_norm (test_b, m);
          if (norm < min_norm)
            {
              min_norm = norm;
              min = iter;
            }
        }
      a = a + n * m;
    }

  if (min_norm < 0)
    {
      *main_block_norm = -1;
      return -2;
    }
  *main_block_norm = min_norm;
  return min;
}


/*
 * Swaping clumns I and J of block matrix A
 * size of A is N * N, size of block is M * M
 * starting from row START
 */
void
swap_massive (double *a, double *b, int n, int m)
{
  double buf;

  for (int i = 0; i < n * m; i++)
    {
      buf = a[i];
      a[i] = b[i];
      b[i] = buf;
    }
}

/*
 * Swaping matrices A and B of size n * m
 */
void
swap_block (double *a, double *b, int n, int m)
{
  int i, j;
  double buf;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          buf = a[i + j * n];
          a[i + j * n] = b[i + j * n];
          b[i + j * n] = buf;
        }
    }
}

//void *
//func (void *arg_local)
//{
//  args *arg;
//  int i, j, k, s, p, n, m, self_number,
//      *pos_j_for_block, min, start, exit_flag, l, *pos_b, swap;
//  double *a, *b, main_block_norm, *c, *block, *a_try, norm, time, t2, t1,
//         *pos_i, *pos_j;

//  pthread_barrier_t *barrier;

//  struct timeval start1, finish1;

//  arg = (args *)arg_local;

//  p = arg->p;
//  self_number = arg->self_number;
//  a = arg->a;
//  b = arg->b;
//  n = arg->n;
//  m = arg->m;
//  norm = arg->norm;
//  barrier = arg->barr;


//  k = n / m;
//  s = n % m;

//  block = new double [m * m];
//  c = new double [m * m];
//  a_try = new double [m * m];
//  pos_j_for_block = new int [m];
//  pos_b = new int [k];

//  for (i = 0; i < k; i++)
//    {
//      pos_b[i] = i;
//    }

//  gettimeofday (&start1, 0);

//  for (i = 0; i < k; i++)
//  {
//      //поиск максимального блока
//      min = search_main_block (a, block, n, m, i, pos_j_for_block, a_try,
//                               norm, &main_block_norm, p, self_number);

//      //СИНХРОНИЗАЦИЯ
//      reduce_main_block (p, main_block_norm, &min, m);

//      if (min < 0)
//        {
//          arg->error = 1;
//          if (self_number == 0)
//            {
//              printf ("Cant invert matrix, step %d\n", i);
//            }

//          delete [] block;
//          delete [] a_try;
//          delete [] pos_j_for_block;
//          delete [] c;
//          delete [] pos_b;

//          return (void *)0;
//        }


//      /*
//       * Swaping columns I and MIN
//       */

//      swap = pos_b[i];
//      pos_b[i] = pos_b[min];
//      pos_b[min] = swap;

//      pos_i = a + pos_of_block (self_number, i, n, m);
//      pos_j = a + pos_of_block (self_number, min, n, m);

//      for (j = self_number; j < k; j += p)
//        {
//          swap_block (pos_i, pos_j, m, m);

//          pos_i += m * m * p;
//          pos_j += m * m * p;
//        }
//      if (j == k && s)
//        {
//          swap_block (pos_i, pos_j, s, m);
//        }


//      //СИНХРОНИЗАЦИЯ
//      pthread_barrier_wait(barrier);

//      cpy_matrix (a_try, a + pos_of_block (i, i, n, m), m, m);
//      gauss (a_try, block, m, pos_j_for_block, norm);

//      start = i + 1 + self_number;

//      //Домножаем строку на обратную матрицу

//      for (j = start; j < k; j += p)
//        {
//          matrix_multiply (block, a + pos_of_block (i, j, n, m), c, m, m, m);
//          cpy_matrix (a + pos_of_block (i, j, n, m), c, m, m);
//        }
//      if (j == k && s)
//        {
//          matrix_multiply (block, a + pos_of_block (i, k, n, m), c, m, m, s);
//          cpy_matrix (a + pos_of_block (i, k, n, m), c, m, s);
//        }

//      for (j = self_number; j < i + 1; j += p)
//        {
//          matrix_multiply (block, b + pos_of_block (i, j, n, m), c, m, m, m);
//          cpy_matrix (b + pos_of_block (i, j, n, m), c, m, m);
//        }
//      if (j == k && s)
//        {
//          matrix_multiply (block, b + pos_of_block (i, k, n, m), c, m, m, s);
//          cpy_matrix (b + pos_of_block (i, k, n, m), c, m, s);
//        }

//      //СИНХРОНИЗАЦИЯ
//      pthread_barrier_wait(barrier);

//      //вычитаем строку i из нижних строк
//      for (j = i + 1; j < k; j++)
//        {
//          cpy_matrix (block, a + pos_of_block (j, i, n, m), m, m);
//          for (l = start; l < k; l += p)
//            {
//              matrix_multiply (block, a + pos_of_block (i, l, n, m), c, m, m, m);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (j, l, n, m), c, m, m);
//            }
//          if (l == k && s)
//            {
//              matrix_multiply (block, a + pos_of_block (i, k, n, m), c, m, m, s);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (j, k, n, m), c, m, s);
//            }

//          for (l = self_number; l < i + 1; l += p)
//            {
//              matrix_multiply (block, b + pos_of_block (i, l, n, m), c, m, m, m);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (j, l, n, m), c, m, m);
//            }
//          if (l == k && s)
//            {
//              matrix_multiply (block, b + pos_of_block (i, l, n, m), c, m, m, s);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (j, l, n, m), c, m, s);
//            }
//        }

//      if (s)
//        {
//          cpy_matrix (block, a + pos_of_block (k, i, n, m), s, m);
//          for (l = start; l < k; l += p)
//            {
//              matrix_multiply (block, a + pos_of_block (i, l, n, m), c, s, m, m);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (k, l, n, m), c, s, m);
//            }
//          if (l == k)
//            {
//              matrix_multiply (block, a + pos_of_block (i, k, n, m), c, s, m, s);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (k, k, n, m), c, s, s);
//            }

//          for (l = self_number; l < i + 1; l += p)
//            {
//              matrix_multiply (block, b + pos_of_block (i, l, n, m), c, s, m, m);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (k, l, n, m), c, s, m);
//            }
//          if (l == k)
//            {
//              matrix_multiply (block, b + pos_of_block (i, k, n, m), c, s, m, s);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (k, k, n, m), c, s, s);
//            }
//        }

//      //СИНХРОНИЗАЦИЯ
//      pthread_barrier_wait(barrier);


//    }

//  exit_flag = 1;

//  //обрабатываем последний блок с последней строчкой
//  if (s)
//    {
//      cpy_matrix (c, a + pos_of_block (k, k, n, m), s, s);
//      if (!gauss (c, block, s, pos_j_for_block, norm))
//        {
//          printf ("Cant invert matrix, step %d\n", i);
//          exit_flag = 0;
//        }


//      for (l = self_number; l < k; l += p)
//        {
//          matrix_multiply (block, b + pos_of_block (k, l, n, m), c, s, s, m);
//          cpy_matrix (b + pos_of_block (k, l, n, m), c, s, m);
//        }
//      if (l == k)
//        {
//          matrix_multiply (block, b + pos_of_block (k, k, n, m), c, s, s, s);
//          cpy_matrix (b + pos_of_block (k, k, n, m), c, s, s);
//        }
//      reduce_last_block (p, &exit_flag);

//      if (exit_flag == 0)
//        {
//          arg->error = 1;

//          delete [] block;
//          delete [] a_try;
//          delete [] pos_j_for_block;
//          delete [] c;
//          delete [] pos_b;

//          return (void *)0;
//        }
//    }

//  //обратный ход
//  if (s)
//    {
//      for (i = k - 1; i >= 0; i--)
//        {
//          cpy_matrix (block, a + pos_of_block (i, k, n, m), m, s);
//          for (j = self_number; j < k; j += p)
//            {
//              matrix_multiply (block, b + pos_of_block (k, j, n, m), c, m, s, m);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (i, j, n, m), c, m, m);
//            }
//          if (j == k)
//            {
//              matrix_multiply (block, b + pos_of_block (k, k, n, m), c, m, s, s);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (i, k, n, m), c, m, s);
//            }
//        }

//      pthread_barrier_wait(barrier);
//    }



//  for (j = k - 1; j >= 0; j--)
//    {
//      for (i = j - 1; i >= 0; i--)
//        {
//          cpy_matrix (block, a + pos_of_block (i, j, n, m), m, m);
//          for (l = self_number; l < k; l += p)
//            {
//              matrix_multiply (block, b + pos_of_block (j, l, n, m), c, m, m, m);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (i, l, n, m), c, m, m);
//            }
//          if (l == k && s)
//            {
//              matrix_multiply (block, b + pos_of_block (j, k, n, m), c, m, m, s);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (i, k, n, m), c, m, s);
//            }
//        }
//      //СИНХРОНИЗАЦИЯ
//      pthread_barrier_wait(barrier);
//    }

//  //обратная перестановка

//  for (i = 0; i < k; i++)
//    {
//      swap = pos_b[i];
//      for (j = self_number; j < k; j += p)
//        {
//          cpy_matrix (a + pos_of_block (swap, j, n, m), b + pos_of_block (i, j, n, m), m, m);
//        }
//      if (j == k)
//        {
//          cpy_matrix (a + pos_of_block (swap, j, n, m), b + pos_of_block (i, j, n, m), m, s);
//        }
//    }

//  //СИНХРОНИЗАЦИЯ
//  pthread_barrier_wait(barrier);

//  for (i = 0; i < k; i++)
//    {
//      for (j = self_number; j < k; j += p)
//        {
//          cpy_matrix (b + pos_of_block (i, j, n, m), a + pos_of_block (i, j, n, m), m, m);
//        }
//      if (j == k)
//        {
//          cpy_matrix (b + pos_of_block (i, j, n, m), a + pos_of_block (i, j, n, m), m, s);
//        }
//    }

//  //СИНХРОНИЗАЦИЯ
//  pthread_barrier_wait(barrier);

//  gettimeofday (&finish1, 0);

//  t1 = start1.tv_sec+(start1.tv_usec/1000000.0);
//  t2 = finish1.tv_sec+(finish1.tv_usec/1000000.0);

//  time = t2 - t1;
//  printf ("Thread %d Elapsed: %f\n", self_number, time);

//  delete [] block;
//  delete [] a_try;
//  delete [] pos_j_for_block;
//  delete [] c;
//  delete [] pos_b;

//  return (void *)1;
//}


// A = A + B
void
matrix_summ (double *a, double *b, int n, int m)
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          a[i + j * n] += b[i + j * n];
        }
    }
}




int gauss_mpi (double *a, double *b, int n, int m, int my_rank, int p, int max_columns, double norm)
{

  int k = n / m;
  int s = n % m;

  double *test_b;
  double *test_block;
  int *test_pos_for_block;
  double local_block_norm;

  test_block = new double [m * m];
  test_b = new double [m * m];
  test_pos_for_block = new int [m];

  double *local_main_block_norm = new double [p];
  int *local_main_block_index = new int [p];

  double *global_main_block_norm = new double [p];
  int *global_main_block_index = new int [p];

  double *min_column = new double [n * m];

  int *index = new int [k];
  for (int i = 0; i < k; i++)
    index[i] = i;

  int buf_int = 0;


  for (int iter = 0; iter < k; iter++)
    {

      mpi_matrix_print (a, n, m, p, my_rank, max_columns);

      //Выяснили глобальный индекс минимального столбца

      memset (local_main_block_norm, 0, p * sizeof (double));
      memset (local_main_block_index, 0, p * sizeof (int));

      memset (global_main_block_norm, 0, p * sizeof (double));
      memset (global_main_block_index, 0, p * sizeof (int));

      int min_number = search_main_block (a, n, m, p, norm, my_rank, iter,
                         test_block, test_b, test_pos_for_block, &local_block_norm);


      local_main_block_norm[my_rank] = local_block_norm;
      local_main_block_index[my_rank] = min_number;

      MPI_Allreduce (local_main_block_norm, global_main_block_norm, p, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce (local_main_block_index, global_main_block_index, p, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      main_block_reduce (global_main_block_norm, global_main_block_index, p, min_number);

      // Отправляем его всем

      memset (min_column, 0, n * m * sizeof(double));

      MPI_Status status;

      if (min_number % p == my_rank)
        {
          copy_massive (min_column, a + n * m * (min_number / p), n * m);
          for (int i = 0; i < p; i++)
            {
              if (i == my_rank)
                continue;

              MPI_Send (min_column, n * m, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

            }
        }
      else
        {
          MPI_Recv (min_column, n * m, MPI_DOUBLE, min_number % p, 0, MPI_COMM_WORLD, &status);
        }

      // Меняем его местами с iter-ым

      buf_int = index[iter];
      index[iter] = index[min_number];
      index[min_number] = buf_int;

      if (iter % p == min_number % p)
        {
          // Оказались в одном процессе, просто физически меняем
          if (min_number % p == my_rank)
            {
              copy_massive (a + n * m * (min_number / p), a + n * m * (iter / p), n * m);
              copy_massive (a + n * m * (iter / p), min_column, n * m);
            }
        }
      else
        {
          // В разных процессах, нужно отправить iter-ый, min_number-ый есть уже у всех
          if (iter % p == my_rank)
            {
              MPI_Send (a + n * m * (iter / p), n * m, MPI_DOUBLE, min_number % p, 0, MPI_COMM_WORLD);

              copy_massive (a + n * m * (iter / p), min_column, n * m);
            }
          if (min_number % p == my_rank)
            {
              MPI_Recv (a + n * m * (min_number / p), n * m, MPI_DOUBLE, iter % p, 0, MPI_COMM_WORLD, &status);
            }
        }

      // Вычисляем обратную матрицу к угловому минимальному элементу

      copy_massive (test_block, min_column + m * m * iter, m * m);
      gauss (test_block, test_b, m, test_pos_for_block, norm);

      // Домножаем строчку на обратную матррицу главного блока

      int start = (iter + 1) / p;

      if (my_rank == k % p && s != 0)
        {
          for (int j = start; j < max_columns - 1; j++)
            {
              matrix_multiply (test_b, a + j * n * m + iter * m * m, test_block, m, m, m);
              cpy_matrix (a + j * n * m + iter * m * m, test_block, m, m);
            }
          matrix_multiply (test_b, a + (max_columns - 1) * n * m + iter * m * s, test_block, m, m, s);
          cpy_matrix (a + (max_columns - 1) * n * m + iter * m * s, test_block, m, s);

        }
      else
        {
          for (int j = start; j < max_columns; j++)
            {
              matrix_multiply (test_b, a + j * n * m + iter * m * m, test_block, m, m, m);
              cpy_matrix (a + j * n * m + iter * m * m, test_block, m, m);
            }
        }


      if (my_rank == k % p && s != 0)
        {
          for (int j = 0; j < (iter + 1) / p; j++)
            {
              matrix_multiply (test_b, b + j * n * m + iter * m * m, test_block, m, m, m);
              cpy_matrix (b + j * n * m + iter * m * m, test_block, m, m);
            }
          matrix_multiply (test_b, b + (max_columns - 1) * n * m + iter * m * s, test_block, m, m, s);
          cpy_matrix (b + (max_columns - 1) * n * m + iter * m * s, test_block, m, s);

        }
      else
        {
          for (int j = 0; j < (iter + 1) / p; j++)
            {
              matrix_multiply (test_b, b + j * n * m + iter * m * m, test_block, m, m, m);
              cpy_matrix (b + j * n * m + iter * m * m, test_block, m, m);
            }
        }
      mpi_matrix_print (test_b, m, 1, p, my_rank, max_columns);
      if (my_rank == 0)
        printf ("min num = %d\n", min_number);

//      // Вычитаем строку из нижних строк
//      for (j = i + 1; j < k; j++)
//        {
//          cpy_matrix (block, a + pos_of_block (j, i, n, m), m, m);
//          for (l = start; l < k; l += p)
//            {
//              matrix_multiply (block, a + pos_of_block (i, l, n, m), c, m, m, m);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (j, l, n, m), c, m, m);
//            }
//          if (l == k && s)
//            {
//              matrix_multiply (block, a + pos_of_block (i, k, n, m), c, m, m, s);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (j, k, n, m), c, m, s);
//            }

//          for (l = self_number; l < i + 1; l += p)
//            {
//              matrix_multiply (block, b + pos_of_block (i, l, n, m), c, m, m, m);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (j, l, n, m), c, m, m);
//            }
//          if (l == k && s)
//            {
//              matrix_multiply (block, b + pos_of_block (i, l, n, m), c, m, m, s);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (j, l, n, m), c, m, s);
//            }
//        }

//      if (s)
//        {
//          cpy_matrix (block, a + pos_of_block (k, i, n, m), s, m);
//          for (l = start; l < k; l += p)
//            {
//              matrix_multiply (block, a + pos_of_block (i, l, n, m), c, s, m, m);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (k, l, n, m), c, s, m);
//            }
//          if (l == k)
//            {
//              matrix_multiply (block, a + pos_of_block (i, k, n, m), c, s, m, s);
//              subtraction_from_matrix_a_matrix_b (a + pos_of_block (k, k, n, m), c, s, s);
//            }

//          for (l = self_number; l < i + 1; l += p)
//            {
//              matrix_multiply (block, b + pos_of_block (i, l, n, m), c, s, m, m);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (k, l, n, m), c, s, m);
//            }
//          if (l == k)
//            {
//              matrix_multiply (block, b + pos_of_block (i, k, n, m), c, s, m, s);
//              subtraction_from_matrix_a_matrix_b (b + pos_of_block (k, k, n, m), c, s, s);
//            }
//        }



    }

  (void)b;
  (void)s;
  (void)k;

  delete test_block;
  delete test_b;
  delete test_pos_for_block;

  delete local_main_block_norm;
  delete local_main_block_index;

  delete global_main_block_norm;
  delete global_main_block_index;

  delete min_column;

  return 0;
}













