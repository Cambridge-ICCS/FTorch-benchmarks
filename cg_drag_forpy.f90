module cg_drag_forpy_mod

use iso_fortran_env, only: error_unit
! Import forpy module for interfacing
use forpy_mod,              only:  import_py, module_py, call_py, object, ndarray, &
                                   forpy_initialize, forpy_finalize, tuple, tuple_create, &
                                   ndarray_create, cast, print_py, dict, dict_create, err_print, &
                                   call_py_noret, list, get_sys_path, ndarray_create_nocopy, &
                                   ndarray_create_empty, ndarray_create_zeros, str, str_create

!-------------------------------------------------------------------

implicit none
private   error_mesg, RADIAN, NOTE, WARNING, FATAL

public    cg_drag_ML_init, cg_drag_ML_end, cg_drag_ML

!--------------------------------------------------------------------
!   data used in this module to bind to forpy
!
!--------------------------------------------------------------------
!   run_emulator    python module
!   paths           python list of strings with system paths
!   model           python 'object' that will contain the model
!   args            python tuple that will contain the model inputs
!   py_pypath       python string
!
!--------------------------------------------------------------------

integer :: ie
type(module_py) :: run_emulator
type(list) :: paths
type(object) :: model
type(tuple) :: args
type(str) :: py_model_dir
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
  !       model_path    full filepath to the model
  !
  !-----------------------------------------------------------------
  character(len=1024), intent(in)        :: model_dir
  character(len=1024), intent(in)        :: model_name
  
  !-----------------------------------------------------------------
  
  ! Initialise the ML model to be used
  ie = forpy_initialize()
  
  ! Add the directory containing forpy related scripts and data to sys.path
  ! This does not appear to work?
  ! export PYTHONPATH=model_dir in the job environment.
  ie = str_create(py_model_dir, trim(model_dir))
  ie = get_sys_path(paths)
  ie = paths%append(py_model_dir)
  
  ! import python modules to `run_emulator`
  ! Note, this will need to be able to load its dependencies
  ! such as `torch`, so you will probably need a venv.
  ie = import_py(run_emulator, trim(model_name))
  if (ie .ne. 0) then
      call err_print
      call error_mesg('cg_drag', 'forpy model not loaded', FATAL)
  end if
  
  ! call initialize function from `run_emulator` python module
  ! loads a trained model to `model`
  ie = call_py(model, run_emulator, "initialize")
  if (ie .ne. 0) then
      call err_print
      call error_mesg('cg_drag', 'call to `initialize` failed', FATAL)
  end if

end subroutine cg_drag_ML_init


!####################################################################

subroutine cg_drag_ML_end

  !-----------------------------------------------------------------
  !    cg_drag_ML_end is called from cg_drag_end and is a destructor
  !    for anything used in the ML part of calculating cg_drag such
  !    as an ML model.
  !
  !-----------------------------------------------------------------
  
  ! destroy the forpy objects
  !
  ! according to forpy no destroy nethod for strings such as 
  ! py_model_dir. Because they are just C under the hood?
  call paths%destroy
  call run_emulator%destroy
  call model%destroy

  call forpy_finalize

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
  
  real, dimension(:,:,:), intent(in)    :: uuu, vvv
  real, dimension(:,:),   intent(in)    :: lat, psfc
  
  real, dimension(:,:,:), intent(out)   :: gwfcng_x, gwfcng_y
  
  !-----------------------------------------------------------------

  !-------------------------------------------------------------------
  !    local variables:
  !
  !       dtdz          temperature lapse rate [ deg K/m ]
  !
  !---------------------------------------------------------------------

  real, dimension(:,:), allocatable, asynchronous  :: uuu_flattened, vvv_flattened
  real, dimension(:,:), allocatable, asynchronous    :: lat_reshaped, psfc_reshaped
  real, dimension(:,:), allocatable, asynchronous  :: gwfcng_x_flattened, gwfcng_y_flattened

  integer :: imax, jmax, kmax, j

  ! forpy variables
  type(ndarray)      :: uuu_nd, vvv_nd, psfc_nd, lat_nd, gwfcng_x_nd, gwfcng_y_nd
  type(tuple)        :: args

  !----------------------------------------------------------------

  ! reshape tensors as required
  imax = size(uuu, 1)
  jmax = size(uuu, 2)
  kmax = size(uuu, 3)

  ! flatten data (nlat, nlon, n) --> (nlat*nlon, n)
  allocate( uuu_flattened(imax*jmax, kmax) )
  allocate( vvv_flattened(imax*jmax, kmax) )
  allocate( lat_reshaped(imax*jmax, 1) )
  allocate( psfc_reshaped(imax*jmax, 1) )
  allocate( gwfcng_x_flattened(imax*jmax, kmax) )
  allocate( gwfcng_y_flattened(imax*jmax, kmax) )

  do j=1,jmax
      uuu_flattened((j-1)*imax+1:j*imax,:) = uuu(:,j,:)
      vvv_flattened((j-1)*imax+1:j*imax,:) = vvv(:,j,:)
      lat_reshaped((j-1)*imax+1:j*imax, 1) = lat(:,j)*RADIAN
      psfc_reshaped((j-1)*imax+1:j*imax, 1) = psfc(:,j)
  end do

  ! creates numpy arrays
  ie = ndarray_create_nocopy(uuu_nd, uuu_flattened)
  ie = ndarray_create_nocopy(vvv_nd, vvv_flattened)
  ie = ndarray_create_nocopy(lat_nd, lat_reshaped)
  ie = ndarray_create_nocopy(psfc_nd, psfc_reshaped)
  ie = ndarray_create_nocopy(gwfcng_x_nd, gwfcng_x_flattened)
  ie = ndarray_create_nocopy(gwfcng_y_nd, gwfcng_y_flattened)

  ! create model input args as tuple
  ie = tuple_create(args,6)
  ie = args%setitem(0,model)
  ie = args%setitem(2,lat_nd)
  ie = args%setitem(3,psfc_nd)
  ie = args%setitem(5,jmax)
  
  ! Zonal
  ie = args%setitem(1,uuu_nd)
  ie = args%setitem(4,gwfcng_x_nd)
  ! Run model and Infer
  ie = call_py_noret(run_emulator, "compute_reshape_drag", args)
  if (ie .ne. 0) then
      call err_print
      call error_mesg('cg_drag_ML', 'inference x call failed', FATAL)
  end if
  
  ! Meridional
  ie = args%setitem(1,vvv_nd)
  ie = args%setitem(4,gwfcng_y_nd)
  ! Run model and Infer
  ie = call_py_noret(run_emulator, "compute_reshape_drag", args)
  if (ie .ne. 0) then
      call err_print
      call error_mesg('cg_drag_ML', 'inference y call failed', FATAL)
  end if


  ! Reshape, and assign to gwfcng
  do j=1,jmax
      gwfcng_x(:,j,:) = gwfcng_x_flattened((j-1)*imax+1:j*imax,:)
      gwfcng_y(:,j,:) = gwfcng_y_flattened((j-1)*imax+1:j*imax,:)
  end do

  ! Cleanup
  call uuu_nd%destroy
  call vvv_nd%destroy
  call psfc_nd%destroy
  call lat_nd%destroy
  call gwfcng_x_nd%destroy
  call gwfcng_y_nd%destroy
  call args%destroy
  
  deallocate( uuu_flattened )
  deallocate( vvv_flattened )
  deallocate( lat_reshaped )
  deallocate( psfc_reshaped )
  deallocate( gwfcng_x_flattened )
  deallocate( gwfcng_y_flattened )


end subroutine cg_drag_ML


!####################################################################

end module cg_drag_forpy_mod
