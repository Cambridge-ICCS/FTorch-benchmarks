program benchmark_stride_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, error_mesg, print_time_stats
  use :: forpy_mod, only: import_py, module_py, call_py, object, ndarray, &
                          forpy_initialize, forpy_finalize, tuple, tuple_create, &
                          ndarray_create, err_print, call_py_noret, list, &
                          get_sys_path, ndarray_create_nocopy, str, str_create

  implicit none

  integer :: i, n
  double precision :: start_time, end_time
  double precision, allocatable :: durations(:)
  real, dimension(:,:), allocatable, asynchronous :: big_array, big_result

  integer :: ie
  type(module_py) :: run_emulator
  type(list) :: paths
  type(object) :: model
  type(tuple) :: args
  type(str) :: py_model_dir
#ifdef USETS
  type(str) :: filename
#endif

  character(len=:), allocatable :: model_dir, model_name
  character(len=128) :: msg
  integer :: ntimes

  type(ndarray) :: big_result_nd, big_array_nd

  print *, "====== FORPY ======"

  call setup(model_dir, model_name, ntimes, n)

  allocate(big_array(n, n))
  allocate(big_result(n, n))
  allocate(durations(ntimes))

  ie = forpy_initialize()
  ie = str_create(py_model_dir, trim(model_dir))
  ie = get_sys_path(paths)
  ie = paths%append(py_model_dir)

  ! import python modules to `run_emulator`
  ie = import_py(run_emulator, trim(model_name))
  if (ie .ne. 0) then
      call err_print
      call error_mesg(__FILE__, __LINE__, "forpy model not loaded")
  end if

#ifdef USETS
  print *, "load torchscript model"
  ! load torchscript saved model
  ie = tuple_create(args,1)
  ie = str_create(filename, trim(model_dir//'/saved_model.pth'))
  ie = args%setitem(0, filename)
  ie = call_py(model, run_emulator, "initialize_ts", args)
  call args%destroy
#else
  print *, "generate model in python runtime"
  ! use python module `run_emulator` to load a trained model
  ie = call_py(model, run_emulator, "initialize")
#endif
  if (ie .ne. 0) then
      call err_print
      call error_mesg(__FILE__, __LINE__, "call to `initialize` failed")
  end if

  do i = 1, ntimes

    call random_number(big_array)

    start_time = omp_get_wtime()

    ! creates numpy arrays
    ie = ndarray_create_nocopy(big_array_nd, big_array)
    ie = ndarray_create_nocopy(big_result_nd, big_result)

    ! create model input args as tuple
    ie = tuple_create(args,3)
    ie = args%setitem(0, model)
    ie = args%setitem(1, big_array_nd)
    ie = args%setitem(2, big_result_nd)

    ie = call_py_noret(run_emulator, "compute", args)
    if (ie .ne. 0) then
        call err_print
        call error_mesg(__FILE__, __LINE__, "inference call failed")
    end if

    ! Clean up.
    call big_result_nd%destroy
    call big_array_nd%destroy
    call args%destroy

    end_time = omp_get_wtime()
    durations(i) = end_time-start_time
    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    big_array(1, 2) = -1.0*big_array(1, 2)
    write(msg, '(A, I8, A, F10.3, A)') "check iteration ", i, " (", durations(i), " s) [omp]"
    call assert(big_array, big_result/2., test_name=msg)
  end do

  call print_time_stats(durations)

  deallocate(big_array)
  deallocate(big_result)
  deallocate(durations)

end program
