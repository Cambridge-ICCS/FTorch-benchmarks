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

    real(dp) :: start_time, end_time, start_loop_time, end_loop_time, mean_loop_time
    real(dp), allocatable :: module_load_durations(:), module_delete_durations(:), tensor_creation_durations(:)
    real(dp), allocatable :: tensor_deletion_durations(:), inference_durations(:), all_durations(:,:)
    character(len=20), allocatable :: messages(:)

    integer :: ie
    type(module_py) :: run_emulator
    type(object) :: model
    type(tuple) :: args

    character(len=:), allocatable :: model_dir, model_name
    character(len=128) :: msg1, msg2, msg3, msg4
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
    allocate(module_load_durations(ntimes))
    allocate(module_delete_durations(ntimes))
    allocate(tensor_creation_durations(ntimes))
    allocate(tensor_deletion_durations(ntimes))
    allocate(inference_durations(ntimes))
    allocate(all_durations(ntimes, 5))
    allocate(messages(5))

    ! Initialise timings with arbitrary large values
    module_load_durations(:) = 100.
    module_delete_durations(:) = 100.
    tensor_creation_durations(:) = 100.
    tensor_deletion_durations(ntimes) = 100.
    inference_durations(ntimes) = 100.
    all_durations(:, :) = 100.
    start_loop_time = 1000.
    end_loop_time = 3000.
    start_time = 1000.
    end_time = 3000.

    if (ntimes .lt. 2) then
      write(*,*) "Error: ntimes must be at least 2"
      return
    end if

    ! Load model (creation/deletion timed at end)
    call load_module(model_dir, model_name, run_emulator, model)

    call load_data(data_file, tensor_length, in_data)

    do i = 1, ntimes

      if (i==2) then
        start_loop_time = omp_get_wtime()
      end if

      ! creates numpy arrays
      ! ------------------------------ Start tensor creation timer ------------------------------
      start_time = omp_get_wtime()
      ie = ndarray_create_nocopy(in_data_nd, in_data)
      ie = ndarray_create_nocopy(out_data_nd, out_data)

      ! create model input args as tuple
      ie = tuple_create(args,3)
      ie = args%setitem(0, model)
      ie = args%setitem(1, in_data_nd)
      ie = args%setitem(2, out_data_nd)
      end_time = omp_get_wtime()
      tensor_creation_durations(i) = end_time - start_time
      ! ------------------------------ End tensor creation timer ------------------------------

      ! ------------------------------ Start inference timer ------------------------------
      start_time = omp_get_wtime()
      ie = call_py_noret(run_emulator, "compute", args)
      end_time = omp_get_wtime()
      inference_durations(i) = end_time - start_time
      ! ------------------------------ End inference timer -----------------------------

      if (ie .ne. 0) then
          call err_print
          call error_mesg(__FILE__, __LINE__, "inference call failed")
      end if

      ! Clean up.
      ! ------------------------------ Start tensor deletion timer ------------------------------
      start_time = omp_get_wtime()
      call out_data_nd%destroy
      call in_data_nd%destroy
      call args%destroy
      end_time = omp_get_wtime()
      tensor_deletion_durations(i) = end_time - start_time
      ! ------------------------------ End tensor deletion timer ------------------------------

      ! Calculate probabilities and output results
      call calc_probs(out_data, probabilities)
      idx = maxloc(probabilities)
      probability = maxval(probabilities)

      ! Check top probability matches expected value
      call assert(probability, expected_prob, test_name="Check probability", rtol_opt=1.0e-5_wp)

      write(msg1, '(A, I10, A, F10.3, A)') "check iteration create tensors", i, " (", tensor_creation_durations(i), " s)"
      write(msg2, '(A, I15, A, F10.3, A)') "check iteration inference", i, " (", inference_durations(i), " s)"
      write(msg3, '(A, I10, A, F10.3, A)') "check iteration delete tensors", i, " (", tensor_deletion_durations(i), " s)"
      print *, trim(msg1)
      print *, trim(msg2)
      print *, trim(msg3)
    end do

    end_loop_time = omp_get_wtime()
    mean_loop_time = (end_loop_time - start_loop_time)/(ntimes - 1)
    write(msg4, '(A, I1, A, F24.4, A)') "Mean time for ", ntimes, " loops", mean_loop_time, " s"
    print *, trim(msg4)

    call time_module(ntimes, model_dir, model_name, module_load_durations, module_delete_durations, run_emulator, model)

    all_durations(:, 1) = module_load_durations
    all_durations(:, 2) = module_delete_durations
    all_durations(:, 3) = tensor_creation_durations
    all_durations(:, 4) = tensor_deletion_durations
    all_durations(:, 5) = inference_durations
    messages = [character(len=20) :: "module creation", "module deletion", "tensor creation", "tensor deletion", "forward pass"]
    call print_all_time_stats(all_durations, messages)

    deallocate(in_data)
    deallocate(out_data)
    deallocate(module_load_durations)
    deallocate(module_delete_durations)
    deallocate(tensor_creation_durations)
    deallocate(tensor_deletion_durations)
    deallocate(inference_durations)
    deallocate(all_durations)
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

  subroutine time_module(ntimes, model_dir, model_name, module_load_durations, module_delete_durations, run_emulator, model)

    implicit none

    integer, intent(in) :: ntimes
    character(len=*), intent(in) :: model_dir, model_name
    real(dp), dimension(:), intent(inout) :: module_load_durations, module_delete_durations
    type(module_py), intent(out) :: run_emulator
    type(object), intent(out) :: model

    integer :: i
    real(dp) :: start_time, end_time

    do i = 1, ntimes
      ! ------------------------------ Start module load timer ------------------------------
      start_time = omp_get_wtime()
      call load_module(model_dir, model_name, run_emulator, model)
      end_time = omp_get_wtime()
      module_load_durations(i) = end_time - start_time
      ! ------------------------------ End module load timer ------------------------------

      ! ------------------------------ Start module deletion timer ------------------------------
      module_delete_durations(i) = 0.
      ! ------------------------------ End module deletion timer ------------------------------
    end do

  end subroutine time_module

  subroutine load_module(model_dir, model_name, run_emulator, model)

    implicit none

    character(len=*), intent(in) :: model_dir, model_name
    type(module_py), intent(out) :: run_emulator
    type(object), intent(out) :: model

    integer :: ie
    type(tuple) :: args
    type(list) :: paths
    type(str) :: py_model_dir
#ifdef USETS
    type(str) :: filename
    print *, "load torchscript model"
#else
    print *, "generate model in python runtime"
#endif

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
    ! load torchscript saved model
    ie = tuple_create(args,1)
    ie = str_create(filename, trim(model_dir//"/"//model_name))
    ie = args%setitem(0, filename)
    ie = call_py(model, run_emulator, "initialize_ts", args)
    call args%destroy
#else
    ! use python module `run_emulator` to load a trained model
    ie = call_py(model, run_emulator, "initialize")
#endif

    if (ie .ne. 0) then
        call err_print
        call error_mesg(__FILE__, __LINE__, "call to `initialize` failed")
    end if

  end subroutine load_module

end program benchmark_resnet
