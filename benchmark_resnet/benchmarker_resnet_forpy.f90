program benchmark_resnet

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, error_mesg, print_all_time_stats
  use :: forpy_mod, only: import_py, module_py, call_py, object, ndarray, &
                          forpy_initialize, forpy_finalize, tuple, tuple_create, &
                          ndarray_create, err_print, call_py_noret, list, &
                          get_sys_path, ndarray_create_nocopy, str, str_create
  use :: precision, only: wp, dp

  implicit none

  call main()

  contains

  subroutine main()

    implicit none

    integer :: i, n
    real(wp), dimension(:,:,:,:), allocatable, asynchronous :: in_data
    real(wp), dimension(:,:), allocatable, asynchronous :: out_data

    real(dp) :: start_time, end_time
    real(dp), allocatable :: durations(:,:)
    character(len=20), allocatable :: messages(:)

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

    type(ndarray) :: out_data_nd, in_data_nd

    ! Binary file containing input tensor
    character(len=*), parameter :: data_file = '../resnet_model/image_tensor.dat'

    ! Length of tensor and number of categories
    integer, parameter :: tensor_length = 150528

    ! Outputs
    integer :: idx(2)
    real(wp), dimension(:,:), allocatable :: probabilities
    real(wp), parameter :: expected_prob = 0.8846225142478943
    real(wp) :: probability

    print *, "====== FORPY ======"

    call setup(model_dir, model_name, ntimes, n)

    allocate(in_data(1, 3, 224, 224))
    allocate(out_data(1, 1000))
    allocate(probabilities(1, 1000))
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
    ie = str_create(filename, trim(model_dir//'/saved_resnet18_model_cpu.pt'))
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

    call load_data(data_file, tensor_length, in_data)

    do i = 1, ntimes

      ! creates numpy arrays
      ! ------------------------------ Start tensor timer ------------------------------
      start_time = omp_get_wtime()
      ie = ndarray_create_nocopy(in_data_nd, in_data)
      ie = ndarray_create_nocopy(out_data_nd, out_data)

      ! create model input args as tuple
      ie = tuple_create(args,3)
      ie = args%setitem(0, model)
      ie = args%setitem(1, in_data_nd)
      ie = args%setitem(2, out_data_nd)
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
      call out_data_nd%destroy
      call in_data_nd%destroy
      call args%destroy
      end_time = omp_get_wtime()
      durations(i, 2) = durations(i, 2) + (end_time - start_time)
      ! ------------------------------ End tensor timer ------------------------------

      ! Calculate probabilities and output results
      call calc_probs(out_data, probabilities)
      idx = maxloc(probabilities)
      probability = maxval(probabilities)

      ! Check top probability matches expected value
      call assert(probability, expected_prob, test_name="Check probability", rtol_opt=1.0e-5_wp)

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

    deallocate(in_data)
    deallocate(out_data)
    deallocate(durations)
    deallocate(messages)
    deallocate(probabilities)

  end subroutine main

  subroutine load_data(filename, tensor_length, in_data)

    implicit none

    character(len=*), intent(in) :: filename
    integer, intent(in) :: tensor_length
    real(wp), dimension(:,:,:,:), intent(out) :: in_data

    real(wp) :: flat_data(tensor_length)
    integer :: ios
    character(len=100) :: ioerrmsg

    ! Read input tensor from Python script
    open(unit=10, file=filename, status='old', access='stream', form='unformatted', action="read", iostat=ios, iomsg=ioerrmsg)
    if (ios /= 0) then
    print *, ioerrmsg
    stop 1
    end if

    read(10, iostat=ios, iomsg=ioerrmsg) flat_data
    if (ios /= 0) then
        print *, ioerrmsg
        stop 1
    end if

    close(10)

    ! Reshape data to tensor input shape
    ! This assumes the data from Python was transposed before saving
    in_data = reshape(flat_data, shape(in_data))

  end subroutine load_data

  subroutine calc_probs(out_data, probabilities)

    implicit none

    real(wp), dimension(:,:), intent(in) :: out_data
    real(wp), dimension(:,:), intent(out) :: probabilities
    real(wp) :: prob_sum

    ! Apply softmax function to calculate probabilties
    probabilities = exp(out_data)
    prob_sum = sum(probabilities)
    probabilities = probabilities / prob_sum

  end subroutine calc_probs

end program benchmark_resnet
