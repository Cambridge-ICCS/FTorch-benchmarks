program benchmark_resnet

  use, intrinsic :: iso_c_binding
  use :: utils, only : assert_real_2d, setup, error_mesg, print_time_stats
  use :: forpy_mod, only: import_py, module_py, call_py, object, ndarray, &
                          forpy_initialize, forpy_finalize, tuple, tuple_create, &
                          ndarray_create, err_print, call_py_noret, list, &
                          get_sys_path, ndarray_create_nocopy, str, str_create

  implicit none

  integer :: i, n
  real :: start_time, end_time
  real, allocatable :: durations(:)
  real, dimension(:,:,:,:), allocatable, asynchronous :: in_data
  real, dimension(:,:), allocatable, asynchronous :: out_data

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

  type(ndarray) :: out_data_nd, in_data_nd

  print *, "====== FORPY ======"

  call setup(model_dir, model_name, ntimes, n)

  allocate(in_data(1, 3, 224, 224))
  allocate(out_data(1, 1000))
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
  ie = str_create(filename, trim(model_dir//'/saved_resnet18_model_cpu.pt'))
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

    in_data = 1.0d0

    call cpu_time(start_time)

    ! creates numpy arrays
    ie = ndarray_create_nocopy(in_data_nd, in_data)
    ie = ndarray_create_nocopy(out_data_nd, out_data)

    ! create model input args as tuple
    ie = tuple_create(args,3)
    ie = args%setitem(0, model)
    ie = args%setitem(1, in_data_nd)
    ie = args%setitem(2, out_data_nd)

    ie = call_py_noret(run_emulator, "compute", args)
    if (ie .ne. 0) then
        call err_print
        call error_mesg(__FILE__, __LINE__, "inference call failed")
    end if

    ! Clean up.
    call out_data_nd%destroy
    call in_data_nd%destroy
    call args%destroy

    call cpu_time(end_time)
    durations(i) = end_time-start_time
    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    write(msg, '(A, I8, A, F10.3, A)') "check iteration ", i, " (", durations(i), " s)"
    print *, trim(msg)
    write (*,*) out_data(1, 1000)
    ! call assert_real_2d(in_data, out_data/2., test_name=msg)
  end do

  call print_time_stats(durations)

  deallocate(in_data)
  deallocate(out_data)
  deallocate(durations)

end program benchmark_resnet
