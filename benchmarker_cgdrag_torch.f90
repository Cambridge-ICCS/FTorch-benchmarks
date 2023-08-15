program benchmark_cgdrag_test

  use, intrinsic :: iso_c_binding
  use :: utils, only : assert_real_2d, setup, print_time_stats
  use :: ftorch

  implicit none

  integer :: i, j, k, ii, jj, kk, n
  real :: start_time, end_time
  real, allocatable :: durations(:)

  integer, parameter :: I_MAX=128, J_MAX=64, K_MAX=40
  real(kind=8), parameter :: PI = 4.0 * ATAN(1.0)
  real(kind=8), parameter :: RADIAN = 180.0 / PI

  real(kind=8), dimension(:,:,:), allocatable, target :: uuu, vvv, gwfcng_x, gwfcng_y
  real(kind=8), dimension(:,:), allocatable, target :: lat, psfc
  integer(c_int), parameter :: n_inputs = 3

  integer(c_int), parameter :: dims_2D = 2
  integer(c_int64_t) :: shape_2D(dims_2D) = [I_MAX*J_MAX, K_MAX]
  integer(c_int) :: stride_2D(dims_2D) = [1,2]
  integer(c_int), parameter :: dims_1D = 2
  integer(c_int64_t) :: shape_1D(dims_1D) = [I_MAX*J_MAX, 1]
  integer(c_int) :: stride_1D(dims_1D) = [1,2]
  integer(c_int), parameter :: dims_out = 2
  integer(c_int64_t) :: shape_out(dims_out) = [I_MAX*J_MAX, K_MAX]
  integer(c_int) :: stride_out(dims_out) = [1,2]

  character(len=:), allocatable :: model_dir, model_name
  character(len=128) :: msg
  integer :: ntimes

  type(torch_module) :: model
  type(torch_tensor), dimension(n_inputs) :: in_tensors
  type(torch_tensor) :: gwfcng_x_tensor, gwfcng_y_tensor

  print *, "====== DIRECT COUPLED ======"

  call setup(model_dir, model_name, ntimes, n)

  allocate(durations(ntimes))

  ! Read gravity wave parameterisation data in from file
  allocate(uuu(I_MAX, J_MAX, K_MAX))
  allocate(vvv(I_MAX, J_MAX, K_MAX))
  allocate(gwfcng_x(I_MAX, J_MAX, K_MAX))
  allocate(gwfcng_y(I_MAX, J_MAX, K_MAX))
  allocate(lat(I_MAX, J_MAX))
  allocate(psfc(I_MAX, J_MAX))
  ! Read in saved input (and output) values
  open(10, file='../input_data/uuu.txt')
  open(11, file='../input_data/vvv.txt')
  open(12, file='../input_data/lat.txt')
  open(13, file='../input_data/psfc.txt')
  do i = 1, I_MAX
      do j = 1, J_MAX
          do k = 1, K_MAX
              read(10, '(3(I4, 1X), E25.16)') ii, jj, kk, uuu(ii,jj,kk)
              read(11, '(3(I4, 1X), E25.16)') ii, jj, kk, vvv(ii,jj,kk)
          end do
          read(12, '(2(I4, 1X), E25.16)') ii, jj, lat(ii,jj)
          read(13, '(2(I4, 1X), E25.16)') ii, jj, psfc(ii,jj)
      end do
  end do

  lat = lat*RADIAN

  model = torch_module_load(model_dir//"/"//model_name//c_null_char)

  do i = 1, ntimes

    call cpu_time(start_time)

    ! Create input and output tensors for the model.
    in_tensors(3) = torch_tensor_from_blob(c_loc(lat), dims_1D, shape_1D, torch_kFloat64, torch_kCPU, stride_1D)
    in_tensors(2) = torch_tensor_from_blob(c_loc(psfc), dims_1D, shape_1D, torch_kFloat64, torch_kCPU, stride_1D)
    
    ! Zonal
    in_tensors(1) = torch_tensor_from_blob(c_loc(uuu), dims_2D, shape_2D, torch_kFloat64, torch_kCPU, stride_2D)
    gwfcng_x_tensor = torch_tensor_from_blob(c_loc(gwfcng_x), dims_out, shape_out, torch_kFloat64, torch_kCPU, stride_out)
    ! Run model and Infer
    call torch_module_forward(model, in_tensors, n_inputs, gwfcng_x_tensor)
    
    ! Meridional
    in_tensors(1) = torch_tensor_from_blob(c_loc(vvv), dims_2D, shape_2D, torch_kFloat64, torch_kCPU, stride_2D)
    gwfcng_y_tensor = torch_tensor_from_blob(c_loc(gwfcng_y), dims_out, shape_out, torch_kFloat64, torch_kCPU, stride_out)
    ! Run model and Infer
    call torch_module_forward(model, in_tensors, n_inputs, gwfcng_y_tensor)

    ! Clean up.
    call torch_tensor_delete(gwfcng_y_tensor)
    call torch_tensor_delete(gwfcng_x_tensor)
    do ii = 1, n_inputs
      call torch_tensor_delete(in_tensors(ii))
    end do

    call cpu_time(end_time)

    durations(i) = end_time-start_time
    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    write(msg, '(A, I8, A, F10.3, A)') "check iteration ", i, " (", durations(i), " s)"
    print *, trim(msg)
    write (*,*) gwfcng_x(1, 1, 1:10)
    write (*,*) gwfcng_y(1, 1, 1:10)
    ! call assert_real_2d(big_array, big_result/2., test_name=msg)
  end do

  call print_time_stats(durations)


  call torch_module_delete(model)

  deallocate(uuu)
  deallocate(vvv)
  deallocate(gwfcng_x)
  deallocate(gwfcng_y)
  deallocate(lat)
  deallocate(psfc)
  deallocate(durations)

end program benchmark_cgdrag_test
