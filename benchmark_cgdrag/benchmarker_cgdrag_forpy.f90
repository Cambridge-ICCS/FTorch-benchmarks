program benchmark_cgdrag_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, error_mesg, print_all_time_stats
  use :: forpy_mod, only: import_py, module_py, call_py, object, ndarray, &
                          forpy_initialize, forpy_finalize, tuple, tuple_create, &
                          ndarray_create, err_print, call_py_noret, list, &
                          get_sys_path, ndarray_create_nocopy, str, str_create
  use :: precision, only: dp

  implicit none

  ! Use double precision, rather than wp defined in precision module
  integer, parameter :: wp = dp

  call main()

  contains

    subroutine main()

    implicit none

    integer :: i, j, k, ii, jj, kk, n
    real(dp) :: start_time, end_time, start_loop_time, end_loop_time, mean_loop_time
    real(dp), allocatable ::   module_load_duration(:), module_delete_durations(:), tensor_creation_durations(:)
    real(dp), allocatable ::   tensor_deletion_durations(:), inference_durations(:), all_durations(:,:)
    character(len=20), allocatable :: messages(:)

    integer, parameter :: I_MAX=128, J_MAX=64, K_MAX=40
    real(wp), parameter :: PI = 4.0 * ATAN(1.0)
    real(wp), parameter :: RADIAN = 180.0 / PI

    real(wp), dimension(:,:,:), allocatable :: uuu, vvv, gwfcng_x, gwfcng_y
    real(wp), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
    real(wp), dimension(:,:), allocatable :: lat, psfc

    real(wp), dimension(:,:), allocatable  :: uuu_flattened, vvv_flattened
    real(wp), dimension(:,:), allocatable  :: lat_reshaped, psfc_reshaped
    real(wp), dimension(:,:), allocatable  :: gwfcng_x_flattened, gwfcng_y_flattened

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
    character(len=128) :: msg1, msg2, msg3, msg4
    integer :: ntimes

    type(ndarray) :: uuu_nd, vvv_nd, gwfcng_x_nd, gwfcng_y_nd, lat_nd, psfc_nd

    print *, "====== FORPY ======"

    call setup(model_dir, model_name, ntimes, n)

    allocate(module_load_duration(ntimes))
    allocate(module_delete_durations(ntimes))
    allocate(tensor_creation_durations(ntimes))
    allocate(tensor_deletion_durations(ntimes))
    allocate(inference_durations(ntimes))
    allocate(all_durations(ntimes, 5))
    allocate(messages(5))

    ! Read gravity wave parameterisation data in from file
    allocate(uuu(I_MAX, J_MAX, K_MAX))
    allocate(vvv(I_MAX, J_MAX, K_MAX))
    allocate(gwfcng_x(I_MAX, J_MAX, K_MAX))
    allocate(gwfcng_y(I_MAX, J_MAX, K_MAX))
    allocate(lat(I_MAX, J_MAX))
    allocate(psfc(I_MAX, J_MAX))

    ! flatten data (nlat, nlon, n) --> (nlat*nlon, n)
    allocate( uuu_flattened(I_MAX*J_MAX, K_MAX) )
    allocate( vvv_flattened(I_MAX*J_MAX, K_MAX) )
    allocate( lat_reshaped(I_MAX*J_MAX, 1) )
    allocate( psfc_reshaped(I_MAX*J_MAX, 1) )
    allocate( gwfcng_x_flattened(I_MAX*J_MAX, K_MAX) )
    allocate( gwfcng_y_flattened(I_MAX*J_MAX, K_MAX) )

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

    ! Read in reference data
    allocate(gwfcng_x_ref(I_MAX, J_MAX, K_MAX))
    allocate(gwfcng_y_ref(I_MAX, J_MAX, K_MAX))
    open(14,file="../cgdrag_model/forpy_reference_x.txt")
    open(15,file="../cgdrag_model/forpy_reference_y.txt")
    read(14,*) gwfcng_x_ref
    read(15,*) gwfcng_y_ref

    close(10)
    close(11)
    close(12)
    close(13)
    close(14)
    close(15)

    ! Initialise timings with arbitrary large values
    module_load_duration(:) = 100.
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
    ie = str_create(filename, trim(model_dir//'/saved_cgdrag_model_cpu.pt'))
    ie = args%setitem(0, filename)
    ie = call_py(model, run_emulator, "initialize_ts", args)
    call args%destroy
#else
    print *, "generate model in python runtime"
    ! use python module `run_emulator` to load a trained model
    ie = call_py(model, run_emulator, "initialize")
#endif

    do j = 1, J_MAX
      uuu_flattened((j-1)*I_MAX+1:j*I_MAX,:) = uuu(:,j,:)
      vvv_flattened((j-1)*I_MAX+1:j*I_MAX,:) = vvv(:,j,:)
      lat_reshaped((j-1)*I_MAX+1:j*I_MAX, 1) = lat(:,j)*RADIAN
      psfc_reshaped((j-1)*I_MAX+1:j*I_MAX, 1) = psfc(:,j)
    end do

    if (ntimes .lt. 2) then
      call err_print
      call error_mesg(__FILE__, __LINE__, "ntimes must be at least 2")
    end if

    do i = 1, ntimes
      if (i==2) then
        start_loop_time = omp_get_wtime()
      end if

      ! creates numpy arrays
      ! ------------------------------ Start tensor creation timer ------------------------------
      start_time = omp_get_wtime()
      ie = ndarray_create_nocopy(uuu_nd, uuu_flattened)
      ie = ndarray_create_nocopy(vvv_nd, vvv_flattened)
      ie = ndarray_create_nocopy(lat_nd, lat_reshaped)
      ie = ndarray_create_nocopy(psfc_nd, psfc_reshaped)
      ie = ndarray_create_nocopy(gwfcng_x_nd, gwfcng_x_flattened)
      ie = ndarray_create_nocopy(gwfcng_y_nd, gwfcng_y_flattened)

      ! create model input args as tuple
      ie = tuple_create(args,6)
      ie = args%setitem(0, model)
      ie = args%setitem(1, uuu_nd)
      ie = args%setitem(2, lat_nd)
      ie = args%setitem(3, psfc_nd)
      ie = args%setitem(4, gwfcng_x_nd)
      ie = args%setitem(5, J_MAX)
      end_time = omp_get_wtime()
      tensor_creation_durations(i) = end_time - start_time
      ! ------------------------------ End tensor creation timer ------------------------------

      ! ------------------------------ Start inference timer ------------------------------
      start_time = omp_get_wtime()
      ie = call_py_noret(run_emulator, "compute_reshape_drag", args)
      end_time = omp_get_wtime()
      inference_durations(i) = end_time - start_time
      ! ------------------------------ End inference timer ------------------------------

      if (ie .ne. 0) then
          call err_print
          call error_mesg(__FILE__, __LINE__, "inference call failed")
      end if

      ! create model input args as tuple
      ! ------------------------------ Start tensor creation timer ------------------------------
      start_time = omp_get_wtime()
      ie = args%setitem(1, vvv_nd)
      ie = args%setitem(4, gwfcng_y_nd)
      end_time = omp_get_wtime()
      tensor_creation_durations(i) = tensor_creation_durations(i) + (end_time - start_time)
      ! ------------------------------ End tensor creation timer ------------------------------

      ! ------------------------------ Start inference timer ------------------------------
      start_time = omp_get_wtime()
      ie = call_py_noret(run_emulator, "compute_reshape_drag", args)
      end_time = omp_get_wtime()
      inference_durations(i) = inference_durations(i) + (end_time - start_time)
      ! ------------------------------ End inference timer ------------------------------

      if (ie .ne. 0) then
          call err_print
          call error_mesg(__FILE__, __LINE__, "inference call failed")
      end if

      ! Reshape, and assign to gwfcng
      do j=1,J_MAX
          gwfcng_x(:,j,:) = gwfcng_x_flattened((j-1)*I_MAX+1:j*I_MAX,:)
          gwfcng_y(:,j,:) = gwfcng_y_flattened((j-1)*I_MAX+1:j*I_MAX,:)
      end do

      ! Clean up.
      ! ------------------------------ Start tensor deletion timer ------------------------------
      start_time = omp_get_wtime()
      call uuu_nd%destroy
      call vvv_nd%destroy
      call gwfcng_x_nd%destroy
      call gwfcng_y_nd%destroy
      call lat_nd%destroy
      call psfc_nd%destroy
      call args%destroy
      end_time = omp_get_wtime()
      tensor_deletion_durations(i) = end_time - start_time
      ! ------------------------------ End tensor deletion timer ------------------------------

      ! Check error
      call assert(gwfcng_x, gwfcng_x_ref, "Check x", rtol_opt=1.0e-8_wp)
      call assert(gwfcng_y, gwfcng_y_ref, "Check y", rtol_opt=1.0e-8_wp)

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

    call time_module(ntimes, model_dir, model_name, module_load_duration, module_delete_durations)

    all_durations(:, 1) = module_load_duration
    all_durations(:, 2) = module_delete_durations
    all_durations(:, 3) = tensor_creation_durations
    all_durations(:, 4) = tensor_deletion_durations
    all_durations(:, 5) = inference_durations
    messages = [character(len=20) :: "module creation", "module deletion", "tensor creation", "tensor deletion", "forward pass"]
    call print_all_time_stats(all_durations, messages)

    deallocate(module_load_duration)
    deallocate(module_delete_durations)
    deallocate(tensor_creation_durations)
    deallocate(tensor_deletion_durations)
    deallocate(inference_durations)
    deallocate(all_durations)
    deallocate(messages)
    deallocate(uuu)
    deallocate(vvv)
    deallocate(gwfcng_x)
    deallocate(gwfcng_y)
    deallocate(lat)
    deallocate(psfc)
    deallocate(uuu_flattened)
    deallocate(vvv_flattened)
    deallocate(lat_reshaped)
    deallocate(psfc_reshaped)
    deallocate(gwfcng_x_flattened)
    deallocate(gwfcng_y_flattened)
    deallocate(gwfcng_x_ref)
    deallocate(gwfcng_y_ref)

    end subroutine main

    subroutine time_module(ntimes, model_dir, model_name, module_load_duration, module_delete_durations)

      implicit none

      integer, intent(in) :: ntimes
      real(dp), dimension(:) :: module_load_duration, module_delete_durations
      integer :: i
      real(dp) :: start_time, end_time
      character(len=*), intent(in) :: model_dir, model_name

      integer :: ie
      type(module_py) :: run_emulator
      type(list) :: paths
      type(object) :: model
      type(tuple) :: args
      type(str) :: py_model_dir
#ifdef USETS
      type(str) :: filename
      print *, "load torchscript model"
#else
      print *, "generate model in python runtime"
#endif

      do i = 1, ntimes
        ! ------------------------------ Start module load timer ------------------------------
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
        ! load torchscript saved model
        ie = tuple_create(args,1)
        ie = str_create(filename, trim(model_dir//'/saved_cgdrag_model_cpu.pt'))
        ie = args%setitem(0, filename)
        ie = call_py(model, run_emulator, "initialize_ts", args)
        call args%destroy
#else
        ! use python module `run_emulator` to load a trained model
        ie = call_py(model, run_emulator, "initialize")
#endif
        end_time = omp_get_wtime()
        module_load_duration(i) = end_time - start_time
        ! ------------------------------ End module load timer ------------------------------

        if (ie .ne. 0) then
            call err_print
            call error_mesg(__FILE__, __LINE__, "call to `initialize` failed")
        end if

        ! ------------------------------ Start module deletion timer ------------------------------
        start_time = omp_get_wtime()
        end_time = omp_get_wtime()
        module_delete_durations(i) = end_time - start_time
        ! ------------------------------ End module deletion timer ------------------------------
      end do

    end subroutine time_module

end program benchmark_cgdrag_test
