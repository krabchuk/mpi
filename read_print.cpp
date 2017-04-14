#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "head.h"

// ВЫБОР ФУНКЦИИ НА СТРОЧКЕ 262

double
f1 (int i, int j)
{
  return 1.0 / (i + j + 1.0);
}

double
f2 (int i, int j)
{
  return fabs (i - j);
}

double
f3 (int i, int j)
{
  return 10 * i + j;
}

int
read_matrix (double *a, int n, int m, FILE *fp)
{
  int i, j, s, k;
  k = n / m;
  s = n - k * m;
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        {
          if (fscanf (fp, "%lf", &a[(i / m) * m * (m * ((((j / m)) / k + 1) % 2) + s * ((j / m) / k))
                      + (j / m) * n * m +
                      (i % m) +
                      (j % m) * (m * ((((i / m)) / k + 1) % 2) + s * ((i / m) / k))]) != 1)
            {
              if (feof (fp))
                return -2;
              else
                return -1;
            }
        }
    }
  return 1;
}

void
generate_matrix (double *a, int n, int m)
{
  int i, j, s, k, pos, block_i, block_j, in_i, in_j;
  k = n / m;
  s = n % m;
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < n; j++)
        {
          block_i = i / m;
          block_j = j / m;
          in_i = i % m;
          in_j = j % m;

          pos = block_j * n * m + in_i;
          if (block_j == k)
            {
              pos += block_i * m * s;
            }
          else
            {
              pos += block_i * m * m;
            }
          if (block_i == k)
            {
              pos += in_j * s;
            }
          else
            {
              pos += in_j * m;
            }

          a[pos] = f2(i, j);
        }
    }
}

void
print_matrix (double *a, int n, int m)
{
  int i, j, s, k, p, pos, block_i, block_j, in_i, in_j;
  k = n / m;
  s = n % m;
  if (n > MAX_SIZE)
    {
      p = MAX_SIZE;
    }
  else
    {
      p = n;
    }
  for (i = 0; i < p; i++)
    {
      for (j = 0; j < p; j++)
        {
          block_i = i / m;
          block_j = j / m;
          in_i = i % m;
          in_j = j % m;

          pos = block_j * n * m + in_i;
          if (block_j == k)
            {
              pos += block_i * m * s;
            }
          else
            {
              pos += block_i * m * m;
            }
          if (block_i == k)
            {
              pos += in_j * s;
            }
          else
            {
              pos += in_j * m;
            }

          printf ("%.3f      ", a[pos]);
        }
      printf ("\n\n");
    }
  printf ("==================================\n");
}

void
print_matrix_real (double *a, int n, int m)
{
  int i, j;
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          printf ("%.3f      ", a[i + j * n]);
        }
      printf ("\n\n");
    }
  printf ("==================================\n");
}

void
print (double *a, int n, int m)
{
  int i, j;

  printf ("==================================\n");
  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          printf ("%e      ", a[i + j * m]);
        }
      printf ("\n\n");
    }
  printf ("==================================\n");
}


int
get_data (char *filename, double *a, double *b, int n, int m, int p, int my_rank)
{
  int i, j;                 // Iterators
  int k;                    // Amount of blocks
  int s;                    // Size of last block

  k = n / m;
  s = n % m;


  if (filename == 0)
    {
      for (i = my_rank; i < k; i += p)
        {
          for (j = 0; j < k; j++)
            {
              set_block (a, j, i, n, m);
              a += m * m;

              if (i == j)
                {
                  set_diag_block (b, m);
                }
              b += m * m;
            }
          if (s)
            {
              set_block (a, j, i, n, m);
              a += m * s;
              b += m * s;
            }
        }
      if (s != 0 && i == k)
        {
          for (j = 0; j < k; j++)
            {
              set_block (a, j, i, n, m);
              a += m * s;
              b += m * s;
            }
          set_block (a, j, i, n, m);
          set_diag_block (b, s);
        }
      return 0;
    }
  else
    {
      if (my_rank == p - 1)
        {
          printf ("This future is still in development\n");
        }
      return 1;
    }

}

/*
 * Setting data for block (GLOBAL_I, GLOBAL_J)
 */
void
set_block (double *a, int global_i, int global_j, int n, int m)
{
  int i, j;                 // Iterators
  int k;                    // Amount of blocks
  int s;                    // Size of last block
  int i_max, j_max, i_start, j_start;

  k = n / m;
  s = n % m;

  j = global_j * m;
  i = global_i * m;

  if (global_i == k)
      i_max = i + s;
  else
      i_max = i + m;

  if (global_j == k)
    j_max = j + s;
  else
    j_max = j + m;

  i_start = i;
  j_start = j;

  for (j = j_start; j < j_max; j++)
    {
      for (i = i_start; i < i_max; i++)
        {
          *a = f2 (i, j);
          a = a + 1;
        }
    }
}

void set_diag_block (double *b, int n)
{
  int i;

  for (i = 0; i < n; i++)
    {
      b[i + i * n] = 1.0;
    }
}






















