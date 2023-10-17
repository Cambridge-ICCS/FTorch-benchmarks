program benchmark_cgdrag_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, print_time_stats
  use :: ftorch
  use :: precision, only: dp

  implicit none

  integer, parameter :: wp = dp

  integer :: i, j, k, ii, jj, kk, n
  real(dp) :: start_time, end_time
  real(dp), allocatable :: durations(:)

  integer, parameter :: I_MAX=128, J_MAX=64, K_MAX=40
  real(wp), parameter :: PI = 4.0 * ATAN(1.0)
  real(wp), parameter :: RADIAN = 180.0 / PI

  real(wp), dimension(:,:,:), allocatable, target :: uuu, vvv, gwfcng_x, gwfcng_y
  real(wp), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
  real(wp), dimension(:,:), allocatable, target :: lat, psfc
  integer(c_int), parameter :: n_inputs = 3

  ! Shape is the shape of the tensor we want to go into the torch
  ! Stride is the mapping between the underlying data and the array
  integer(c_int), parameter :: dims_2D = 2
  integer(c_int64_t) :: shape_2D(dims_2D) = [I_MAX*J_MAX, K_MAX]
  integer(c_int) :: stride_2D(dims_2D) = [1,2]
  integer(c_int), parameter :: dims_1D = 2
  integer(c_int64_t) :: shape_1D(dims_1D) = [I_MAX*J_MAX, 1]
  integer(c_int) :: stride_1D(dims_1D) = [1,2]
  integer(c_int), parameter :: dims_out = 2
  integer(c_int64_t) :: shape_out(dims_out) = [I_MAX*J_MAX, K_MAX]
  integer(c_int) :: stride_out(dims_out) = [1,2]

  real(wp), dimension(:,:), allocatable, target :: uuu_flattened, vvv_flattened
  real(wp), dimension(:,:), allocatable, target :: lat_reshaped, psfc_reshaped

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
  open(10, file='../cgdrag_model/uuu.txt')
  open(11, file='../cgdrag_model/vvv.txt')
  open(12, file='../cgdrag_model/lat.txt')
  open(13, file='../cgdrag_model/psfc.txt')
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

  model = torch_module_load(model_dir//"/"//model_name)

  ! flatten data (nlat, nlon, n) --> (nlat*nlon, n)
  allocate( uuu_flattened(I_MAX*J_MAX, K_MAX) )
  allocate( vvv_flattened(I_MAX*J_MAX, K_MAX) )
  allocate( lat_reshaped(I_MAX*J_MAX, 1) )
  allocate( psfc_reshaped(I_MAX*J_MAX, 1) )

  allocate(gwfcng_x_ref(I_MAX, J_MAX, K_MAX))
  allocate(gwfcng_y_ref(I_MAX, J_MAX, K_MAX))

  open(10,file="../cgdrag_model/forpy_reference_x.txt")
  open(20,file="../cgdrag_model/forpy_reference_y.txt")

  read(10,*) gwfcng_x_ref
  read(20,*) gwfcng_y_ref

  close(10)
  close(20)

  do i = 1, ntimes

    do j=1,J_MAX
        uuu_flattened((j-1)*I_MAX+1:j*I_MAX,:) = uuu(:,j,:)
        vvv_flattened((j-1)*I_MAX+1:j*I_MAX,:) = vvv(:,j,:)
        lat_reshaped((j-1)*I_MAX+1:j*I_MAX, 1) = lat(:,j)
        psfc_reshaped((j-1)*I_MAX+1:j*I_MAX, 1) = psfc(:,j)
    end do


    ! Create input and output tensors for the model.
    in_tensors(3) = torch_tensor_from_blob(c_loc(lat_reshaped), dims_1D, shape_1D, torch_kFloat64, torch_kCPU, stride_1D)
    in_tensors(2) = torch_tensor_from_blob(c_loc(psfc_reshaped), dims_1D, shape_1D, torch_kFloat64, torch_kCPU, stride_1D)

    ! Zonal
    in_tensors(1) = torch_tensor_from_blob(c_loc(uuu_flattened), dims_2D, shape_2D, torch_kFloat64, torch_kCPU, stride_2D)
    gwfcng_x_tensor = torch_tensor_from_blob(c_loc(gwfcng_x), dims_out, shape_out, torch_kFloat64, torch_kCPU, stride_out)
    ! Run model and Infer
    call torch_module_forward(model, in_tensors, n_inputs, gwfcng_x_tensor)

    ! Meridional
    in_tensors(1) = torch_tensor_from_blob(c_loc(vvv_flattened), dims_2D, shape_2D, torch_kFloat64, torch_kCPU, stride_2D)
    gwfcng_y_tensor = torch_tensor_from_blob(c_loc(gwfcng_y), dims_out, shape_out, torch_kFloat64, torch_kCPU, stride_out)

    ! Run model and Infer
    start_time = omp_get_wtime()
    call torch_module_forward(model, in_tensors, n_inputs, gwfcng_y_tensor)
    end_time = omp_get_wtime()

    ! Clean up.
    call torch_tensor_delete(gwfcng_y_tensor)
    call torch_tensor_delete(gwfcng_x_tensor)
    do ii = 1, n_inputs
      call torch_tensor_delete(in_tensors(ii))
    end do

    durations(i) = end_time-start_time
    ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
    write(msg, '(A, I8, A, F10.3, A)') "check iteration ", i, " (", durations(i), " s) [omp]"
    print *, trim(msg)

    ! Check error
    call assert(gwfcng_x, gwfcng_x_ref, "Check x", rtol_opt=1.0e-8_wp)
    call assert(gwfcng_y, gwfcng_y_ref, "Check y", rtol_opt=1.0e-8_wp)

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
  deallocate(gwfcng_x_ref)
  deallocate(gwfcng_y_ref)

end program benchmark_cgdrag_test
