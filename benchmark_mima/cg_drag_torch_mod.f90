module cg_drag_torch_mod

! Imports primitives used to interface with C
use, intrinsic :: iso_c_binding, only: c_int64_t, c_loc
! Import library for interfacing with PyTorch
use ftorch
use :: precision, only: dp

!-------------------------------------------------------------------

implicit none

! Use double precision, rather than wp defined in precision module
integer, parameter :: wp = dp

private   RADIAN

public    cg_drag_ML_init, cg_drag_ML_end, cg_drag_ML

!--------------------------------------------------------------------
!   data used in this module to bind to FTorch
!
!--------------------------------------------------------------------
!   model    ML model type bound to python
!
!--------------------------------------------------------------------

type(torch_module) :: model

real(wp), parameter :: PI = 4.0 * ATAN(1.0)
real(wp), parameter :: RADIAN = 180.0 / PI

!--------------------------------------------------------------------
!--------------------------------------------------------------------

contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!                      PUBLIC SUBROUTINES
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


!####################################################################

subroutine cg_drag_ML_init(model_dir, model_name)

  !-----------------------------------------------------------------
  !    cg_drag_ML_init is called from cg_drag_init and initialises
  !    anything required for the ML calculation of cg_drag such as
  !    an ML model
  !
  !-----------------------------------------------------------------

  !-----------------------------------------------------------------
  !    intent(in) variables:
  !
  !       model_dir    full filepath to the model directory
  !       model_name   filename of the TorchScript model
  !
  !-----------------------------------------------------------------
  character(len=1024), intent(in)        :: model_dir
  character(len=1024), intent(in)        :: model_name

  !-----------------------------------------------------------------

  ! Initialise the ML model to be used
  model = torch_module_load(trim(model_dir)//"/"//trim(model_name))

end subroutine cg_drag_ML_init


!####################################################################

subroutine cg_drag_ML_end

  !-----------------------------------------------------------------
  !    cg_drag_ML_end is called from cg_drag_end and is a destructor
  !    for anything used in the ML part of calculating cg_drag such
  !    as an ML model.
  !
  !-----------------------------------------------------------------

  ! destroy the model
  call torch_module_delete(model)

end subroutine cg_drag_ML_end


!####################################################################

subroutine cg_drag_ML(uuu, vvv, psfc, lat, gwfcng_x, gwfcng_y)

  !-----------------------------------------------------------------
  !    cg_drag_ML returns the x and y gravity wave drag forcing
  !    terms following calculation using an external neural net.
  !
  !-----------------------------------------------------------------

  !-----------------------------------------------------------------
  !    intent(in) variables:
  !
  !       is,js    starting subdomain i,j indices of data in
  !                the physics_window being integrated
  !       uuu,vvv  arrays of model u and v wind
  !       psfc     array of model surface pressure
  !       lat      array of model latitudes at cell boundaries [radians]
  !
  !    intent(out) variables:
  !
  !       gwfcng_x time tendency for u eqn due to gravity-wave forcing
  !                [ m/s^2 ]
  !       gwfcng_y time tendency for v eqn due to gravity-wave forcing
  !                [ m/s^2 ]
  !
  !-----------------------------------------------------------------

  real(wp), dimension(:,:,:), target, intent(in)    :: uuu, vvv
  real(wp), dimension(:,:), target,   intent(in)    :: psfc
  real(wp), dimension(:,:), target                  :: lat

  real(wp), dimension(:,:,:), intent(out), target   :: gwfcng_x, gwfcng_y

  !-----------------------------------------------------------------

  !-------------------------------------------------------------------
  !    local variables:
  !
  !       dtdz          temperature lapse rate [ deg K/m ]
  !
  !---------------------------------------------------------------------

  integer :: imax, jmax, kmax

  integer(c_int), parameter :: dims_2D = 2
  integer(c_int64_t) :: shape_2D(dims_2D)
  integer(c_int) :: stride_2D(dims_2D)
  integer(c_int), parameter :: dims_1D = 2
  integer(c_int64_t) :: shape_1D(dims_1D)
  integer(c_int) :: stride_1D(dims_1D)
  integer(c_int), parameter :: dims_out = 2
  integer(c_int64_t) :: shape_out(dims_out)
  integer(c_int) :: stride_out(dims_out)

  ! Set up types of input and output data and the interface with C
  type(torch_tensor) :: gwfcng_x_tensor, gwfcng_y_tensor
  integer(c_int), parameter :: n_inputs = 3
  type(torch_tensor), dimension(n_inputs), target :: model_input_arr

  !----------------------------------------------------------------

  ! reshape tensors as required
  imax = size(uuu, 1)
  jmax = size(uuu, 2)
  kmax = size(uuu, 3)

  ! pseudo-flatten data (nlat, nlon, n) --> (nlat*nlon, n)
  ! Note that the '1D' tensor has 2 dimensions, one of which is size 1
  shape_2D = (/ imax*jmax, kmax /)
  shape_1D = (/ imax*jmax, 1 /)
  shape_out = (/ imax*jmax, kmax /)

  stride_1D = (/ 1, 2 /)
  stride_2D =  (/ 1, 2 /)
  stride_out =  (/ 1, 2 /)

  lat = lat*RADIAN

  ! Create input/output tensors from the above arrays
  model_input_arr(3) = torch_tensor_from_blob(c_loc(lat), dims_1D, shape_1D, torch_kFloat64, torch_kCPU, stride_1D)
  model_input_arr(2) = torch_tensor_from_blob(c_loc(psfc), dims_1D, shape_1D, torch_kFloat64, torch_kCPU, stride_1D)

  ! Zonal
  model_input_arr(1) = torch_tensor_from_blob(c_loc(uuu), dims_2D, shape_2D, torch_kFloat64, torch_kCPU, stride_2D)
  gwfcng_x_tensor = torch_tensor_from_blob(c_loc(gwfcng_x), dims_out, shape_out, torch_kFloat64, torch_kCPU, stride_out)
  ! Run model and Infer
  call torch_module_forward(model, model_input_arr, n_inputs, gwfcng_x_tensor)

  ! Meridional
  model_input_arr(1) = torch_tensor_from_blob(c_loc(vvv), dims_2D, shape_2D, torch_kFloat64, torch_kCPU, stride_2D)
  gwfcng_y_tensor = torch_tensor_from_blob(c_loc(gwfcng_y), dims_out, shape_out, torch_kFloat64, torch_kCPU, stride_out)
  ! Run model and Infer
  call torch_module_forward(model, model_input_arr, n_inputs, gwfcng_y_tensor)

  write (*,*) gwfcng_x(1, 1, 1:10)
  write (*,*) gwfcng_y(1, 1, 1:10)
  ! Cleanup
  call torch_tensor_delete(model_input_arr(1))
  call torch_tensor_delete(model_input_arr(2))
  call torch_tensor_delete(model_input_arr(3))
  call torch_tensor_delete(gwfcng_x_tensor)
  call torch_tensor_delete(gwfcng_y_tensor)

  ! write(*,*) gwfcng_y(1:5, 1:5, 1)

end subroutine cg_drag_ML


!####################################################################

end module cg_drag_torch_mod
