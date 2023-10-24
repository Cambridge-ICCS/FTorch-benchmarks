program benchmark_stride_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, error_mesg, print_all_time_stats
  use :: forpy_mod, only: import_py, module_py, call_py, object, ndarray, &
                          forpy_initialize, forpy_finalize, tuple, tuple_create, &
                          ndarray_create, err_print, call_py_noret, list, &
                          get_sys_path, ndarray_create_nocopy, str, str_create
  use :: precision, only: wp, dp

  implicit none

  integer :: i, n
  real(dp) :: start_time, end_time
  real(dp), allocatable :: durations(:,:)
  character(len=20), allocatable :: messages(:)
  real(wp), dimension(:,:), allocatable, asynchronous :: big_array, big_result

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
  character(len=128) :: msg1, msg2
  integer :: ntimes

  type(ndarray) :: big_result_nd, big_array_nd

  print *, "====== FORPY ======"

  call setup(model_dir, model_name, ntimes, n)

  allocate(big_array(n, n))
  allocate(big_result(n, n))
  allocate(durations(ntimes, 3))
  allocate(messages(3))

  ! ------------------------------ Start module timer ------------------------------
  start_time = omp_get_wtime()
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
  ie = str_create(filename, trim(model_dir//'/saved_model.pt'))
  ie = args%setitem(0, filename)
  ie = call_py(model, run_emulator, "initialize_ts", args)
  call args%destroy
#else
  print *, "generate model in python runtime"
  ! use python module `run_emulator` to load a trained model
  ie = call_py(model, run_emulator, "initialize")
#endif
  end_time = omp_get_wtime()
  durations(:, 1) = end_time - start_time
  ! ------------------------------ End module timer ------------------------------

  if (ie .ne. 0) then
      call err_print
      call error_mesg(__FILE__, __LINE__, "call to `initialize` failed")
  end if

  do i = 1, ntimes

    call random_number(big_array)

    ! creates numpy arrays
    ! ------------------------------ Start tensor timer ------------------------------
    start_time = omp_get_wtime()
    ie = ndarray_create_nocopy(big_array_nd, big_array)
    ie = ndarray_create_nocopy(big_result_nd, big_result)

    ! create model input args as tuple
    ie = tuple_create(args,3)
    ie = args%setitem(0, model)
    ie = args%setitem(1, big_array_nd)
    ie = args%setitem(2, big_result_nd)
    end_time = omp_get_wtime()
    durations(i, 2) = end_time - start_time
    ! ------------------------------ End tensor timer ------------------------------

    ! ------------------------------ Start inference timer ------------------------------
    start_time = omp_get_wtime()
    ie = call_py_noret(run_emulator, "compute", args)
    end_time = omp_get_wtime()
    durations(i, 3) = end_time - start_time
    ! ------------------------------ End inference timer ------------------------------

    if (ie .ne. 0) then
      call err_print
      call error_mesg(__FILE__, __LINE__, "inference call failed")
    end if

    ! Clean up.
    ! ------------------------------ Start tensor timer ------------------------------
    start_time = omp_get_wtime()
    call big_result_nd%destroy
    call big_array_nd%destroy
    call args%destroy
    end_time = omp_get_wtime()
    durations(i, 2) = durations(i, 2) + (end_time - start_time)
    ! ------------------------------ End tensor timer ------------------------------

    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    big_array(1, 2) = -1.0*big_array(1, 2)
    call assert(big_array, big_result/2., test_name="Check array")

      ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    write(msg1, '(A, I8, A, F10.3, A)') "check iteration inference", i, " (", durations(i, 3), " s) [omp]"
    write(msg2, '(A, I8, A, F10.3, A)') "check iteration tensors", i, " (", durations(i, 2), " s) [omp]"
    print *, trim(msg1)
    print *, trim(msg2)
  end do

  ! ------------------------------ Start module timer ------------------------------
  start_time = omp_get_wtime()
  end_time = omp_get_wtime()
  durations(:, 1) = durations(:, 1) + (end_time - start_time)
  ! ------------------------------ End module timer ------------------------------

  messages = [character(len=20) :: "--- modules ---", "--- tensors ---", "--- forward pass ---"]
  call print_all_time_stats(durations, messages)

  deallocate(big_array)
  deallocate(big_result)
  deallocate(durations)
  deallocate(messages)

end program
