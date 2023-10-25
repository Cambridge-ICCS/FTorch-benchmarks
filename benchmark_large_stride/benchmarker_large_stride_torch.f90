program benchmark_stride_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, print_all_time_stats
  use :: ftorch
  use :: precision, only: wp, dp

  implicit none

  integer, parameter :: torch_wp = torch_kFloat32

  integer :: i, ii, n
  real(dp) :: start_time, end_time
  real(dp), allocatable :: durations(:,:)
  character(len=20), allocatable :: messages(:)
  real(wp), dimension(:,:), allocatable, target :: big_array, big_result

  integer(c_int), parameter :: n_inputs = 1
  integer(c_int64_t) :: shape_2d(2)
  integer(c_int) :: stride_2d(2)

  character(len=:), allocatable :: model_dir, model_name
  character(len=128) :: msg1, msg2
  integer :: ntimes

  type(torch_tensor) :: result_tensor
  type(torch_tensor), dimension(n_inputs), target :: input_array
  type(torch_module) :: model

  print *, "====== DIRECT COUPLED ======"

  call setup(model_dir, model_name, ntimes, n)

  allocate(big_array(n, n))
  allocate(big_result(n, n))
  allocate(durations(ntimes, 3))
  allocate(messages(3))

  ! ------------------------------ Start module timer ------------------------------
  start_time = omp_get_wtime()
  model = torch_module_load(model_dir//"/"//model_name)
  end_time = omp_get_wtime()
  durations(:, 1) = end_time - start_time
  ! ------------------------------ End module timer ------------------------------

  shape_2d = (/ n, n /)
  stride_2d = (/ 1, 2 /)

  do i = 1, ntimes

    call random_number(big_array)

    ! Create input and output tensors for the model.
    ! ------------------------------ Start tensor timer ------------------------------
    start_time = omp_get_wtime()
    input_array(1) = torch_tensor_from_blob(c_loc(big_array), 2, shape_2d, torch_wp, torch_kCPU, stride_2d)
    result_tensor = torch_tensor_from_blob(c_loc(big_result), 2, shape_2d, torch_wp, torch_kCPU, stride_2d)
    end_time = omp_get_wtime()
    durations(i, 2) = end_time - start_time
    ! ------------------------------ End tensor timer ------------------------------

    ! ------------------------------ Start inference timer ------------------------------
    start_time = omp_get_wtime()
    call torch_module_forward(model, input_array, n_inputs, result_tensor)
    end_time = omp_get_wtime()
    durations(i, 3) = end_time - start_time
    ! ------------------------------ End inference timer ------------------------------

    ! Clean up.
    ! ------------------------------ Start tensor timer ------------------------------
    start_time = omp_get_wtime()
    call torch_tensor_delete(result_tensor)
    do ii = 1, n_inputs
      call torch_tensor_delete(input_array(ii))
    end do
    end_time = omp_get_wtime()
    durations(i, 2) = durations(i, 2) + (end_time - start_time)
    ! ------------------------------ End tensor timer ------------------------------

    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    big_array(1, 2) = -1.0*big_array(1, 2)
    call assert(big_array, big_result/2., test_name="Check array")

    write(msg1, '(A, I8, A, F10.3, A)') "check iteration inference", i, " (", durations(i, 3), " s) [omp]"
    write(msg2, '(A, I8, A, F10.3, A)') "check iteration tensors", i, " (", durations(i, 2), " s) [omp]"
    print *, trim(msg1)
    print *, trim(msg2)
  end do

  ! ------------------------------ Start module timer ------------------------------
  start_time = omp_get_wtime()
  call torch_module_delete(model)
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
