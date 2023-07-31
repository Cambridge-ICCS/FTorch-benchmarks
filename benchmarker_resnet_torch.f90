program benchmark_resnet_test

  use, intrinsic :: iso_c_binding
  use :: utils, only : assert_real_2d, setup, print_time_stats
  use :: ftorch

  implicit none

  integer :: i, ii, n
  real :: start_time, end_time
  real, allocatable :: durations(:)

  real(c_float), dimension(:,:,:,:), allocatable, target :: in_data
  integer(c_int), parameter :: n_inputs = 1
  real(c_float), dimension(:,:), allocatable, target :: out_data

  integer(c_int), parameter :: in_dims = 4
  integer(c_int64_t) :: in_shape(in_dims) = [1, 3, 224, 224]
  integer(c_int) :: in_layout(in_dims) = [1,2,3,4]
  integer(c_int), parameter :: out_dims = 2
  integer(c_int64_t) :: out_shape(out_dims) = [1, 1000]
  integer(c_int) :: out_layout(out_dims) = [1,2]

  character(len=:), allocatable :: model_dir, model_name
  character(len=128) :: msg
  integer :: ntimes

  type(torch_module) :: model
  type(torch_tensor), dimension(1) :: in_tensor
  type(torch_tensor) :: out_tensor

  print *, "====== DIRECT COUPLED ======"

  call setup(model_dir, model_name, ntimes, n)

  allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
  allocate(out_data(out_shape(1), out_shape(2)))
  allocate(durations(ntimes))

  model = torch_module_load(model_dir//"/"//model_name)

  do i = 1, ntimes

    ! Initialise data
    in_data = 1.0d0

    call cpu_time(start_time)

    ! Create input and output tensors for the model.
    in_tensor(1) = torch_tensor_from_blob(c_loc(in_data), in_dims, in_shape, torch_kFloat32, torch_kCPU, in_layout)
    out_tensor = torch_tensor_from_blob(c_loc(out_data), out_dims, out_shape, torch_kFloat32, torch_kCPU, out_layout)

    call torch_module_forward(model, in_tensor, n_inputs, out_tensor)

    ! Clean up.
    call torch_tensor_delete(out_tensor)
    do ii = 1, n_inputs
      call torch_tensor_delete(in_tensor(ii))
    end do

    call cpu_time(end_time)

    durations(i) = end_time-start_time
    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    write(msg, '(A, I8, A, F10.3, A)') "check iteration ", i, " (", durations(i), " s)"
    print *, trim(msg)
    write (*,*) out_data(1, 1000)
    ! call assert_real_2d(big_array, big_result/2., test_name=msg)
  end do

  call print_time_stats(durations)


  call torch_module_delete(model)

  deallocate(in_data)
  deallocate(out_data)
  deallocate(durations)

end program benchmark_resnet_test
