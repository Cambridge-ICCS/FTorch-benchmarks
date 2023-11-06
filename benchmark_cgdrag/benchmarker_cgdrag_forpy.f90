program benchmark_cgdrag_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, error_mesg, print_time_stats, print_all_time_stats
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

      integer :: i, j, n
      real(dp) :: start_time, end_time, start_loop_time, end_loop_time
      real(dp), dimension(:), allocatable :: module_load_durations, module_delete_durations, loop_durations, inference_durations
      real(dp), dimension(:), allocatable :: allocation_durations, deallocation_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), dimension(:,:), allocatable :: all_durations
      character(len=20), dimension(:), allocatable :: messages

      integer, parameter :: I_MAX=128, J_MAX=64, K_MAX=40

      real(wp), dimension(:,:,:), allocatable :: uuu, vvv, gwfcng_x, gwfcng_y
      real(wp), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
      real(wp), dimension(:,:), allocatable :: lat, psfc

      real(wp), dimension(:,:), allocatable  :: uuu_flattened, vvv_flattened
      real(wp), dimension(:,:), allocatable  :: lat_reshaped, psfc_reshaped
      real(wp), dimension(:,:), allocatable  :: gwfcng_x_flattened, gwfcng_y_flattened

      integer :: ie
      type(module_py) :: run_emulator
      type(object) :: model
      type(tuple) :: args

      character(len=:), allocatable :: model_dir, model_name
      character(len=128) :: msg1, msg2, msg3, msg4, msg5, msg6
      integer :: ntimes

      type(ndarray) :: uuu_nd, vvv_nd, gwfcng_x_nd, gwfcng_y_nd, lat_nd, psfc_nd

      ! Set flag to .true. via command line argument --alloc_in_loop
      ! to allocate/deallocate flattened arrays during each loop. Default (.false.) is set in setup().
      logical :: alloc_in_loop

      print *, "====== FORPY ======"

      call setup(model_dir, model_name, ntimes, n, alloc_in_loop)
      if (ntimes .lt. 2) then
        write(*,*) "Error: ntimes must be at least 2"
        return
      end if

      ! Allocate arrays shared with FTorch implementation and read in data
      call init_common_arrays(ntimes, I_MAX, J_MAX, K_MAX, uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, &
                              module_load_durations, module_delete_durations, loop_durations, allocation_durations, deallocation_durations, &
                              tensor_creation_durations, tensor_deletion_durations, inference_durations, all_durations, messages, &
                              start_loop_time, end_loop_time, start_time, end_time)

      ! Reshape arrays, if not done for every loop
      if (.not. alloc_in_loop) then
        call init_reshaped_arrays(I_MAX, J_MAX, K_MAX, uuu, vvv, lat, psfc, uuu_flattened, vvv_flattened, &
                            lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
      end if

      ! Load model (creation/deletion timed at end)
#ifdef USETS
      print *, "load torchscript model"
#else
      print *, "generate model in python runtime"
#endif
      call load_module(model_dir, model_name, run_emulator, model)

      do i = 1, ntimes

        ! ------------------------------ Start loop timer ----------------------------
        start_loop_time = omp_get_wtime()

        ! ------------------------------ Start allocation timer ----------------------------
        start_time = omp_get_wtime()
        if (alloc_in_loop) then
          call init_reshaped_arrays(I_MAX, J_MAX, K_MAX, uuu, vvv, lat, psfc, uuu_flattened, vvv_flattened, &
          lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
        end if
        end_time = omp_get_wtime()
        allocation_durations(i) = end_time - start_time
        ! ------------------------------ End allocation timer ----------------------------

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

        ! ------------------------------ Start inference timer ------------------------------
        ! Include with inference, as necessary for useful output
        start_time = omp_get_wtime()
        ! Reshape, and assign to gwfcng
        do j = 1, J_MAX
            gwfcng_x(:, j, :) = gwfcng_x_flattened((j - 1) * I_MAX + 1:j * I_MAX, :)
            gwfcng_y(:, j, :) = gwfcng_y_flattened((j - 1) * I_MAX + 1:j * I_MAX, :)
        end do
        end_time = omp_get_wtime()
        inference_durations(i) = inference_durations(i) + (end_time - start_time)
        ! ------------------------------ End inference timer ------------------------------

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
        call assert(gwfcng_x, gwfcng_x_ref, "Check x", rtol_opt=1.0e-7_wp)
        call assert(gwfcng_y, gwfcng_y_ref, "Check y", rtol_opt=1.0e-7_wp)

        ! ------------------------------ Start deallocation timer ------------------------------
        start_time = omp_get_wtime()
        if (alloc_in_loop) then
          call deallocate_reshaped_arrays(uuu_flattened, vvv_flattened, lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
        end if
        end_time = omp_get_wtime()
        deallocation_durations(i) = end_time - start_time
        ! ------------------------------ End deallocation timer ------------------------------

        end_loop_time = omp_get_wtime()
        loop_durations(i) = end_loop_time - start_loop_time
        ! ------------------------------ End loop timer ----------------------------

        write(msg1, '(A, I18, A, F11.6, A)') "check iteration inference", i, " (", inference_durations(i), " s)"
        write(msg2, '(A, I13, A, F11.6, A)') "check iteration create tensors", i, " (", tensor_creation_durations(i), " s)"
        write(msg3, '(A, I13, A, F11.6, A)') "check iteration delete tensors", i, " (", tensor_deletion_durations(i), " s)"
        write(msg4, '(A, I12, A, F11.6, A)') "check iteration allocate arrays", i, " (", allocation_durations(i), " s)"
        write(msg5, '(A, I10, A, F11.6, A)') "check iteration deallocate arrays", i, " (", deallocation_durations(i), " s)"
        write(msg6, '(A, I18, A, F11.6, A)') "check iteration full loop", i, " (", loop_durations(i), " s)"
        print *, trim(msg1)
        print *, trim(msg2)
        print *, trim(msg3)
        print *, trim(msg4)
        print *, trim(msg5)
        print *, trim(msg6)

      end do

      call time_module(ntimes, model_dir, model_name, module_load_durations, module_delete_durations, run_emulator, model)

      ! Call individual print for loop, to avoid adding to combined mean
      call print_time_stats(loop_durations, "full loop")

      all_durations(:, 1) = module_load_durations
      all_durations(:, 2) = module_delete_durations
      all_durations(:, 3) = allocation_durations
      all_durations(:, 4) = deallocation_durations
      all_durations(:, 5) = tensor_creation_durations
      all_durations(:, 6) = tensor_deletion_durations
      all_durations(:, 7) = inference_durations
      messages = [character(len=20) :: "module creation", "module deletion", "array allocation", "array deallocation", &
                  "tensor creation", "tensor deletion", "forward pass"]
      call print_all_time_stats(all_durations, messages)

      call deallocate_common_arrays(uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, module_load_durations, &
                                    module_delete_durations, loop_durations, allocation_durations, deallocation_durations, &
                                    tensor_creation_durations, tensor_deletion_durations, inference_durations, all_durations, messages)

      if (.not. alloc_in_loop) then
        call deallocate_reshaped_arrays(uuu_flattened, vvv_flattened, lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
      end if

    end subroutine main

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
        ! We can only call forpy_finalize once
        if (i == ntimes) then
          start_time = omp_get_wtime()
          call forpy_finalize
          end_time = omp_get_wtime()
          module_delete_durations(:) = (end_time - start_time) / (ntimes + 1)
        end if
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
      ie = str_create(filename, trim(model_dir//"/"//"saved_cgdrag_model_cpu.pt"))
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

    subroutine init_common_arrays(ntimes, I_MAX, J_MAX, K_MAX, uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, &
                                  module_load_durations, module_delete_durations, loop_durations, allocation_durations, &
                                  deallocation_durations, tensor_creation_durations, tensor_deletion_durations, inference_durations, &
                                  all_durations, messages, start_loop_time, end_loop_time, start_time, end_time)

      implicit none

      integer, intent(in):: ntimes, I_MAX, J_MAX, K_MAX

      real(wp), intent(out), dimension(:,:,:), allocatable :: uuu, vvv, gwfcng_x, gwfcng_y
      real(wp), intent(out), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
      real(wp), intent(out), dimension(:,:), allocatable :: lat, psfc

      real(dp), intent(out), dimension(:), allocatable :: module_load_durations, module_delete_durations, loop_durations, inference_durations
      real(dp), intent(out), dimension(:), allocatable :: allocation_durations, deallocation_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), intent(out), dimension(:,:), allocatable :: all_durations
      character(len=20), intent(out), dimension(:), allocatable :: messages

      real(dp), intent(out) :: start_loop_time, end_loop_time, start_time, end_time

      real(wp), parameter :: PI = 4.0 * ATAN(1.0)
      real(wp), parameter :: RADIAN = 180.0 / PI

      integer :: i, j, k, ii, jj, kk

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
            read(10, '(3(I4, 1X), E25.16)') ii, jj, kk, uuu(ii, jj, kk)
            read(11, '(3(I4, 1X), E25.16)') ii, jj, kk, vvv(ii, jj, kk)
            end do
          read(12, '(2(I4, 1X), E25.16)') ii, jj, lat(ii, jj)
          read(13, '(2(I4, 1X), E25.16)') ii, jj, psfc(ii, jj)
        end do
      end do

      lat = lat * RADIAN

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

      ! Allocate arrays for timings
      allocate(module_load_durations(ntimes))
      allocate(module_delete_durations(ntimes))
      allocate(loop_durations(ntimes))
      allocate(allocation_durations(ntimes))
      allocate(deallocation_durations(ntimes))
      allocate(tensor_creation_durations(ntimes))
      allocate(tensor_deletion_durations(ntimes))
      allocate(inference_durations(ntimes))
      allocate(all_durations(ntimes, 7))
      allocate(messages(7))

      ! Initialise timings with arbitrary large values
      module_load_durations(:) = 100.
      module_delete_durations(:) = 100.
      loop_durations(:) = 100.
      allocation_durations(:) = 100.
      deallocation_durations(:) = 100.
      tensor_creation_durations(:) = 100.
      tensor_deletion_durations(ntimes) = 100.
      inference_durations(ntimes) = 100.
      all_durations(:, :) = 100.
      start_loop_time = 1000.
      end_loop_time = 3000.
      start_time = 1000.
      end_time = 3000.

    end subroutine init_common_arrays

    subroutine init_reshaped_arrays(I_MAX, J_MAX, K_MAX, uuu, vvv, lat, psfc, uuu_flattened, vvv_flattened, &
                              lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)

      implicit none

      integer, intent(in):: I_MAX, J_MAX, K_MAX
      real(wp), intent(in), dimension(:,:,:), allocatable :: uuu, vvv
      real(wp), intent(in), dimension(:,:), allocatable :: lat, psfc

      real(wp), intent(out), dimension(:,:), allocatable :: uuu_flattened, vvv_flattened
      real(wp), intent(out), dimension(:,:), allocatable :: lat_reshaped, psfc_reshaped
      real(wp), intent(out), dimension(:,:), allocatable :: gwfcng_x_flattened, gwfcng_y_flattened

      integer :: j

      ! flatten data (nlat, nlon, n) --> (nlat*nlon, n)
      allocate(uuu_flattened(I_MAX * J_MAX, K_MAX))
      allocate(vvv_flattened(I_MAX * J_MAX, K_MAX))
      allocate(lat_reshaped(I_MAX * J_MAX, 1))
      allocate(psfc_reshaped(I_MAX * J_MAX, 1))
      allocate(gwfcng_x_flattened(I_MAX * J_MAX, K_MAX))
      allocate(gwfcng_y_flattened(I_MAX * J_MAX, K_MAX))

      do j = 1, J_MAX
        uuu_flattened((j - 1) * I_MAX + 1:j * I_MAX, :) = uuu(:, j, :)
        vvv_flattened((j - 1) * I_MAX + 1:j * I_MAX, :) = vvv(:, j, :)
        lat_reshaped((j - 1) * I_MAX + 1:j * I_MAX, 1) = lat(:, j)
        psfc_reshaped((j - 1) * I_MAX + 1:j * I_MAX, 1) = psfc(:, j)
      end do

    end subroutine init_reshaped_arrays

    subroutine deallocate_common_arrays(uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, module_load_durations, &
                                        module_delete_durations, loop_durations, allocation_durations, deallocation_durations, &
                                        tensor_creation_durations, tensor_deletion_durations, inference_durations, all_durations, messages)

      implicit none

      real(dp), intent(inout), dimension(:), allocatable :: module_load_durations, module_delete_durations, loop_durations, inference_durations
      real(dp), intent(inout), dimension(:), allocatable :: allocation_durations, deallocation_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), intent(inout), dimension(:,:), allocatable :: all_durations
      character(len=20), intent(inout), dimension(:), allocatable :: messages

      real(wp), intent(inout), dimension(:,:,:), allocatable :: uuu, vvv, gwfcng_x, gwfcng_y
      real(wp), intent(inout), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
      real(wp), intent(inout), dimension(:,:), allocatable :: lat, psfc

      deallocate(module_load_durations)
      deallocate(module_delete_durations)
      deallocate(loop_durations)
      deallocate(allocation_durations)
      deallocate(deallocation_durations)
      deallocate(tensor_creation_durations)
      deallocate(tensor_deletion_durations)
      deallocate(inference_durations)
      deallocate(all_durations)
      deallocate(messages)
      deallocate(uuu)
      deallocate(vvv)
      deallocate(gwfcng_x)
      deallocate(gwfcng_y)
      deallocate(gwfcng_x_ref)
      deallocate(gwfcng_y_ref)
      deallocate(lat)
      deallocate(psfc)

    end subroutine deallocate_common_arrays

    subroutine deallocate_reshaped_arrays(uuu_flattened, vvv_flattened, lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)

      implicit none

      real(wp), intent(inout), dimension(:,:), allocatable :: uuu_flattened, vvv_flattened
      real(wp), intent(inout), dimension(:,:), allocatable :: lat_reshaped, psfc_reshaped
      real(wp), intent(inout), dimension(:,:), allocatable :: gwfcng_x_flattened, gwfcng_y_flattened

      deallocate(uuu_flattened)
      deallocate(vvv_flattened)
      deallocate(lat_reshaped)
      deallocate(psfc_reshaped)
      deallocate(gwfcng_x_flattened)
      deallocate(gwfcng_y_flattened)

    end subroutine deallocate_reshaped_arrays

end program benchmark_cgdrag_test
