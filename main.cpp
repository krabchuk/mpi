#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


#include "head.h"
#include "mpi_functions.h"




int
main (int argc, char *argv[])
{
  //kekek
  //char *filename;                     //Name of data file
  int n;                              // Size of matrix
  int m;                              // Size of block
  int p;                              // Amount of processes
  int my_rank;                        // Number of process in group
  int max_columns;                   // Amount of columns in one process
  int k;                              // Amount of blocks
  double norm = 0;                    // Norm of original matrix
  int s;                              // Size of last block
 // FILE *fp;                           // File descriptor
  double *a;                          // Original matrix
  double *b;                          // Attached matrix
  double *c;
  double block_norm;
  double *test_block;
  double *test_b;
  int *test_pos_for_block;
  int iter = 1;

  MPI_Init  (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &p);


  /* Initializing of n and m */
  if (my_rank == 0)
    {

      if (argc != 4 && argc != 3)
        {
          printf ("Usage: %s n m [filename]\n", argv[0]);
          MPI_Finalize ();
          return 0;
        }

      if (sscanf (argv[1], "%d", &n) != 1)
        {
          printf ("Usage: %s n m [filename]\n", argv[0]);
          MPI_Finalize ();
          return 0;
        }

      if (n <= 0)
        {
          printf ("Wrong n value\n");
          MPI_Finalize ();
          return 0;
        }

      if (sscanf (argv[2], "%d", &m) != 1)
        {
          printf ("Usage: %s n m [filename]\n", argv[0]);
          MPI_Finalize ();
          return 0;
        }

      if (m <= 0 || m > n)
        {
          printf ("Wrong m value\n");
          MPI_Finalize ();
          return 0;
        }

    }
  else
    {

      if (argc != 4 && argc != 3)
        {
          MPI_Finalize ();
          return 0;
        }

      if (sscanf (argv[1], "%d", &n) != 1)
        {
          MPI_Finalize ();
          return 0;
        }

      if (n <= 0)
        {
          MPI_Finalize ();
          return 0;
        }

      if (sscanf (argv[2], "%d", &m) != 1)
        {
          MPI_Finalize ();
          return 0;
        }

      if (m <= 0 || m > n)
        {
          MPI_Finalize ();
          return 0;
        }

    }


  s = n % m;
  k = n / m;

  if (s != 0)
    {
      if ((k + 1) % p != 0)
        {
          max_columns = ((k + 1)/ p) + 1;
        }
      else
        {
          max_columns = (k + 1) / p;
        }
    }
  else
    {
      if (k % p != 0)
        {
          max_columns = (k / p) + 1;
        }
      else
        {
          max_columns = k / p;
        }
    }

  int total_elements_amount = 0;

  if (s != 0)
    {
      total_elements_amount = (k + 1) * m * m * max_columns;

      a = new double [total_elements_amount];
      b = new double [total_elements_amount];
      c = new double [total_elements_amount];

      memset (a, 0, total_elements_amount * sizeof (double));
      memset (b, 0, total_elements_amount * sizeof (double));
      memset (c, 0, total_elements_amount * sizeof (double));
    }
  else
    {
      total_elements_amount = n * m * max_columns;

      a = new double [total_elements_amount];
      b = new double [total_elements_amount];
      c = new double [total_elements_amount];

      memset (a, 0, total_elements_amount * sizeof (double));
      memset (b, 0, total_elements_amount * sizeof (double));
      memset (c, 0, total_elements_amount * sizeof (double));
    }


  if (argc == 4)
    {
      matrix_read_mpi (argv[3], a, n, m, p, my_rank, max_columns);
      init_e_mpi (b, n, m, p, my_rank);
    }
  else
    {
      get_data (0, a, b, n, m, p, my_rank);
    }

//  matrix_read_mpi (argv[3], a, n, m, p, my_rank, max_columns);
  memset (b, 0, total_elements_amount * sizeof (double));
  matrix_read_mpi (argv[3], b, n, m, p, my_rank, max_columns);


  matrix_mult_mpi (a, b, c, n, m, p, my_rank);

  for (int i = 0; i < total_elements_amount; i++)
    {
      printf ("my rank = %d c[%d] = %f\n", my_rank, i, c[i]);
    }

//  norm = matrix_norm_mpi (a, n, m, p, my_rank);
(void)norm;
//  gauss_mpi (a, b, n, m, my_rank, p, norm);


  delete a;
  delete b;
  delete c;
  MPI_Finalize ();
  (void)block_norm;
  (void)test_block;
  (void)test_b;
  (void)test_pos_for_block;
  (void)iter;
  (void)k;
  (void)s;
  return 0;
}
/*
  a = new double [n * n];
  b = new double [n * n];

  if (argc == 5)
    {
      filename = argv[4];
      fp = fopen (filename, "r");
      if (!fp)
        {
          printf ("Cant open %s\n", filename);
          delete [] a;
          delete [] b;
          return 0;
        }
      res = read_matrix (a, n, m, fp);
      if (res < 0)
        {
          printf ("Error %d\n", res);
          fclose (fp);
          delete [] a;
          delete [] b;
          return 0;
        }
    }
  else
    {
      generate_matrix (a, n, m);
    }

  init (b, n, m);
  norm = matrix_norm (a, n, m);

  pthread_barrier_init (&barrier, 0, p);

  arguments = new args [p];

  for (i = 0; i < p; i++)
    {
      arguments[i].a = a;
      arguments[i].b = b;
      arguments[i].m = m;
      arguments[i].n = n;
      arguments[i].norm = norm;
      arguments[i].p = p;
      arguments[i].self_number = i;
      arguments[i].barr = &barrier;
    }

  //print_matrix (a, n, m);
  printf ("\n\nN = %d M = %d\n", n, m);
  gettimeofday (&start, 0);

  for (i = 1; i < p; i++)
    {
      if (pthread_create (&t_id, 0, func, arguments + i) != 0)
        {
          printf ("Can not create thread %d\n", i);
          abort ();
        }
    }
  func (arguments + 0);

  if (arguments[0].error)
    {
      delete [] arguments;
      delete [] a;
      delete [] b;

      return 0;
    }

  gettimeofday (&finish, 0);

  t1 = start.tv_sec+(start.tv_usec/1000000.0);
  t2 = finish.tv_sec+(finish.tv_usec/1000000.0);

  time = t2 - t1;
  printf ("Elapsed: %f\n", time);

  //print_matrix (b, n, m);

  if (argc == 5)
    {
      filename = argv[4];
      fp = fopen (filename, "r");
      if (!fp)
        {
          printf ("Cant open %s\n", filename);

          delete [] arguments;
          delete [] a;
          delete [] b;

          return 0;
        }
      res = read_matrix (a, n, m, fp);
      if (res < 0)
        {
          printf ("Error %d\n", res);
          fclose (fp);

          delete [] arguments;
          delete [] a;
          delete [] b;

          return 0;
        }
    }
  else
    {
      generate_matrix (a, n, m);
    }


  c = new double [n * n];

  for (i = 0; i < p; i++)
    {
      arguments[i].c = c;
    }

  gettimeofday (&start, 0);

  for (i = 1; i < p; i++)
    {
      if (pthread_create (&t_id, 0, matrix_multiply_parallel, arguments + i) != 0)
        {
          printf ("Can not create thread %d\n", i);
          abort ();
        }
    }
  matrix_multiply_parallel (arguments + 0);

  subtract_e (c, n, m);

  norm = matrix_norm (c, n, m);

  gettimeofday (&finish, 0);

  t1 = start.tv_sec+(start.tv_usec/1000000.0);
  t2 = finish.tv_sec+(finish.tv_usec/1000000.0);

  time = t2 - t1;
  printf ("Elapsed norm: %f\n", time);

  printf ("Norm: %e\n", norm);

  delete [] arguments;
  delete [] a;
  delete [] b;
  delete [] c;

  return 1;
}*/
