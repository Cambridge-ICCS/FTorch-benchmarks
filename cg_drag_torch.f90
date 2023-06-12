module cg_drag_torch_mod

use iso_fortran_env, only: error_unit
! Imports primitives used to interface with C
use, intrinsic :: iso_c_binding, only: c_int64_t, c_float, c_char, c_null_char, c_ptr, c_loc
! Import library for interfacing with PyTorch
use ftorch




!-------------------------------------------------------------------

implicit none
private   error_mesg, RADIAN, NOTE, WARNING, FATAL

public    cg_drag_ML_init, cg_drag_ML_end, cg_drag_ML

!--------------------------------------------------------------------
!   data used in this module to bind to FTorch
!
!--------------------------------------------------------------------
!   model    ML model type bound to python
!
!--------------------------------------------------------------------

type(torch_module) :: model

integer, parameter :: NOTE=0, WARNING=1, FATAL=2
real(kind=8), parameter :: PI = 4.0 * ATAN(1.0)
real(kind=8), parameter :: RADIAN = 180.0 / PI

!--------------------------------------------------------------------
!--------------------------------------------------------------------

contains

! PRIVATE ROUTINES
 subroutine error_mesg (routine, message, level)
  character(len=*), intent(in) :: routine, message
  integer,          intent(in) :: level

!  input:
!      routine   name of the calling routine (character string)
!      message   message written to output   (character string)
!      level     set to NOTE, MESSAGE, or FATAL (integer)

    write(error_unit, '(a,":", a)') routine, message

 end subroutine error_mesg

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
  model = torch_module_load(trim(model_dir)//"/"//trim(model_name)//c_null_char)

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
  
  real(kind=8), dimension(:,:,:), intent(in)    :: uuu, vvv
  real(kind=8), dimension(:,:),   intent(in)    :: lat, psfc
  
  real(kind=8), dimension(:,:,:), intent(out), target   :: gwfcng_x, gwfcng_y
  
  !-----------------------------------------------------------------

  !-------------------------------------------------------------------
  !    local variables:
  !
  !       dtdz          temperature lapse rate [ deg K/m ]
  !
  !---------------------------------------------------------------------

  real(kind=8), dimension(:,:), allocatable, target  :: uuu_reshaped, vvv_reshaped
  real(kind=8), dimension(:,:), allocatable, target    :: lat_reshaped, psfc_reshaped
  real(kind=8), dimension(:,:), allocatable, target  :: gwfcng_x_reshaped, gwfcng_y_reshaped

  integer :: imax, jmax, kmax, j, k

  integer(c_int), parameter :: dims_2D = 2
  integer(c_int64_t) :: shape_2D(dims_2D)
  integer(c_int), parameter :: dims_1D = 2
  integer(c_int64_t) :: shape_1D(dims_1D)
  integer(c_int), parameter :: dims_out = 2
  integer(c_int64_t) :: shape_out(dims_out)

  ! Set up types of input and output data and the interface with C
  type(torch_tensor) :: gwfcng_x_tensor, gwfcng_y_tensor
  integer(c_int), parameter :: n_inputs = 3
  type(torch_tensor), dimension(n_inputs), target :: model_input_arr
  
  !----------------------------------------------------------------

  ! reshape tensors as required
  imax = size(uuu, 1)
  jmax = size(uuu, 2)
  kmax = size(uuu, 3)

  ! Note that the '1D' tensor has 2 dimensions, one of which is size 1
  shape_2D = (/ imax*jmax, kmax /)
  shape_1D = (/ imax*jmax, 1 /)
  shape_out = (/ imax*jmax, kmax /)

  ! flatten data (nlat, nlon, n) --> (nlat*nlon, n)
  allocate( uuu_reshaped(kmax, imax*jmax) )
  allocate( vvv_reshaped(kmax, imax*jmax) )
  allocate( lat_reshaped(1, imax*jmax) )
  allocate( psfc_reshaped(1, imax*jmax) )
  allocate( gwfcng_x_reshaped(kmax, imax*jmax) )
  allocate( gwfcng_y_reshaped(kmax, imax*jmax) )

  do j=1,jmax
      do k=1, kmax
          uuu_reshaped(k, (j-1)*imax+1:j*imax) = uuu(:,j,k)
          vvv_reshaped(k, (j-1)*imax+1:j*imax) = vvv(:,j,k)
      end do
      lat_reshaped(1, (j-1)*imax+1:j*imax) = lat(:,j)*RADIAN
      psfc_reshaped(1, (j-1)*imax+1:j*imax) = psfc(:,j)
  end do

  ! Create input/output tensors from the above arrays
  model_input_arr(3) = torch_tensor_from_blob(c_loc(lat_reshaped), dims_1D, shape_1D, torch_kFloat64, torch_kCPU)
  model_input_arr(2) = torch_tensor_from_blob(c_loc(psfc_reshaped), dims_1D, shape_1D, torch_kFloat64, torch_kCPU)
  
  ! Zonal
  model_input_arr(1) = torch_tensor_from_blob(c_loc(uuu_reshaped), dims_2D, shape_2D, torch_kFloat64, torch_kCPU)
  gwfcng_x_tensor = torch_tensor_from_blob(c_loc(gwfcng_x_reshaped), dims_out, shape_out, torch_kFloat64, torch_kCPU)
  ! Run model and Infer
  call torch_module_forward(model, model_input_arr, n_inputs, gwfcng_x_tensor)
  
  ! Meridional
  model_input_arr(1) = torch_tensor_from_blob(c_loc(vvv_reshaped), dims_2D, shape_2D, torch_kFloat64, torch_kCPU)
  gwfcng_y_tensor = torch_tensor_from_blob(c_loc(gwfcng_y_reshaped), dims_out, shape_out, torch_kFloat64, torch_kCPU)
  ! Run model and Infer
  call torch_module_forward(model, model_input_arr, n_inputs, gwfcng_y_tensor)


  ! Convert back into fortran types, reshape, and assign to gwfcng
  do j=1,jmax
      do k=1, kmax
          gwfcng_x(:,j,k) = gwfcng_x_reshaped(k, (j-1)*imax+1:j*imax)
          gwfcng_y(:,j,k) = gwfcng_y_reshaped(k, (j-1)*imax+1:j*imax)
      end do
  end do

  ! Cleanup
  call torch_tensor_delete(model_input_arr(1))
  call torch_tensor_delete(model_input_arr(2))
  call torch_tensor_delete(model_input_arr(3))
  call torch_tensor_delete(gwfcng_x_tensor)
  call torch_tensor_delete(gwfcng_y_tensor)
  deallocate( uuu_reshaped )
  deallocate( vvv_reshaped )
  deallocate( lat_reshaped )
  deallocate( psfc_reshaped )
  deallocate( gwfcng_x_reshaped )
  deallocate( gwfcng_y_reshaped )


end subroutine cg_drag_ML


!####################################################################

end module cg_drag_torch_mod
