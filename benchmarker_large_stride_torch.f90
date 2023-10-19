program benchmark_stride_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, print_time_stats
  use :: ftorch

  implicit none

  integer :: i, ii, n
  double precision :: start_time, end_time
  double precision, allocatable :: durations(:)
  real, dimension(:,:), allocatable, target :: big_array, big_result

  integer(c_int), parameter :: n_inputs = 1
  integer(c_int64_t) :: shape_2d(2)
  integer(c_int) :: stride_2d(2)

  character(len=:), allocatable :: model_dir, model_name
  character(len=128) :: msg
  integer :: ntimes, input_device
  logical :: use_cuda = .false.

  type(torch_tensor) :: result_tensor
  type(torch_tensor), dimension(n_inputs), target :: input_array
  type(torch_module) :: model

  print *, "====== DIRECT COUPLED ======"

  call setup(model_dir, model_name, ntimes, n, use_cuda)

  if (use_cuda) then
    input_device = torch_kCUDA
  else
    input_device = torch_kCPU
  end if

  allocate(big_array(n, n))
  allocate(big_result(n, n))
  allocate(durations(ntimes))

  model = torch_module_load(model_dir//"/"//model_name)

  shape_2d = (/ n, n /)
  stride_2d = (/ 1, 2 /)

  do i = 1, ntimes

    call random_number(big_array)

    start_time = omp_get_wtime()

    ! Create input and output tensors for the model.
    input_array(1) = torch_tensor_from_blob(c_loc(big_array), 2, shape_2d, torch_kFloat32, input_device, stride_2d)
    result_tensor = torch_tensor_from_blob(c_loc(big_result), 2, shape_2d, torch_kFloat32, torch_kCPU, stride_2d)

    call torch_module_forward(model, input_array, n_inputs, result_tensor)

    ! Clean up.
    call torch_tensor_delete(result_tensor)
    do ii = 1, n_inputs
      call torch_tensor_delete(input_array(ii))
    end do

    end_time = omp_get_wtime()

    durations(i) = end_time-start_time
    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    big_array(1, 2) = -1.0*big_array(1, 2)
    write(msg, '(A, I8, A, F10.3, A)') "check iteration ", i, " (", durations(i), " s) [omp]"
    call assert(big_array, big_result/2., test_name=msg)
  end do

  call print_time_stats(durations)


  call torch_module_delete(model)

  deallocate(big_array)
  deallocate(big_result)
  deallocate(durations)

end program
