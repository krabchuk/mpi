#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>


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


  int total_elements_amount = n * m * max_columns;

  a = new double [total_elements_amount];
  b = new double [total_elements_amount];
  c = new double [total_elements_amount];

  memset (a, 0, total_elements_amount * sizeof (double));
  memset (b, 0, total_elements_amount * sizeof (double));
  memset (c, 0, total_elements_amount * sizeof (double));

  int error = 0;
  int global_error = 0;


  if (argc == 4)
    {
      mpi_matrix_read (argv[3], a, n, m, p, my_rank, max_columns, error);
      mpi_init_e (b, n, m, p, my_rank);
      MPI_Allreduce (&error, &global_error, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (global_error)
        {
          delete [] a;
          delete [] b;
          delete [] c;
          MPI_Finalize ();
          return 0;
        }
    }
  else
    {
      get_data (0, a, b, n, m, p, my_rank);
    }

  norm = mpi_matrix_norm (a, n, m, p, my_rank, max_columns);

  mpi_matrix_print (a, n, m, p, my_rank, max_columns);

  struct timeval start, finish;

  double t1, t2, time;

  if (my_rank == 0)
    gettimeofday (&start, 0);

  if (gauss_mpi (a, b, n, m, my_rank, p, max_columns, norm))
    {
      delete [] a;
      delete [] b;
      delete [] c;
      MPI_Finalize ();

      return 0;
    }

  if (my_rank == 0)
    {
      gettimeofday (&finish, 0);

      t1 = start.tv_sec+(start.tv_usec/1000000.0);
      t2 = finish.tv_sec+(finish.tv_usec/1000000.0);

      time = t2 - t1;
      printf ("Elapsed: %f\n", time);
    }

  if (p == 1 && n > 1000)
    {
      delete [] a;
      delete [] b;
      delete [] c;
      MPI_Finalize ();

      return 0;
    }

  mpi_matrix_print (b, n, m, p, my_rank, max_columns);

  if (argc == 4)
    {
      mpi_matrix_read (argv[3], a, n, m, p, my_rank, max_columns, error);
    }
  else
    {
      get_data (0, a, c, n, m, p, my_rank);
    }

  memset (c, 0, total_elements_amount * sizeof (double));

  mpi_matrix_mult (a, b, c, n, m, p, my_rank);

  mpi_subtract_e (c, n, m, p, my_rank, max_columns);

  norm = mpi_matrix_norm (c, n, m, p, my_rank, max_columns);

  if (my_rank == 0)
    {
      printf ("Residual = %e\n", norm);
    }


  delete [] a;
  delete [] b;
  delete [] c;
  MPI_Finalize ();

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
