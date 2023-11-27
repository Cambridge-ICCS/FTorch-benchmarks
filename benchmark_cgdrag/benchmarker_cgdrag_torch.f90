program benchmark_cgdrag_test

  use, intrinsic :: iso_c_binding, only : c_loc, c_int, c_int64_t
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, print_time_stats, print_all_time_stats
  use :: ftorch
  use :: precision, only: dp

  implicit none

  ! Use double precision, rather than wp defined in precision module
  integer, parameter :: wp = dp
  integer, parameter :: torch_wp = torch_kFloat64

  call main()

  contains

    subroutine main()

      implicit none

      integer :: i, j, n, ii
      real(dp) :: start_time, end_time, start_loop_time, end_loop_time
      real(dp), dimension(:), allocatable :: loop_durations, inference_durations, allocation_durations
      real(dp), dimension(:), allocatable :: deallocation_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), dimension(:,:), allocatable :: all_durations
      character(len=20), dimension(:), allocatable :: messages

      integer, parameter :: I_MAX=128, J_MAX=64, K_MAX=40

      real(wp), dimension(:,:,:), allocatable, target :: uuu, vvv, gwfcng_x, gwfcng_y
      real(wp), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
      real(wp), dimension(:,:), allocatable, target :: lat, psfc

      real(wp), dimension(:,:), allocatable, target :: uuu_flattened, vvv_flattened
      real(wp), dimension(:,:), allocatable, target :: lat_reshaped, psfc_reshaped
      real(wp), dimension(:,:), allocatable, target  :: gwfcng_x_flattened, gwfcng_y_flattened

      integer, parameter :: n_inputs = 3

      integer, parameter :: dims_1D = 2
      integer, parameter :: dims_2D = 2
      integer, parameter :: dims_out = 2
      integer :: shape_2D(dims_2D) = [I_MAX * J_MAX, K_MAX]
      integer :: stride_2D(dims_2D) = [1, 2]
      integer :: shape_1D(dims_1D) = [I_MAX * J_MAX, 1]
      integer :: stride_1D(dims_1D) = [1, 2]
      integer :: shape_out(dims_out) = [I_MAX * J_MAX, K_MAX]
      integer :: stride_out(dims_out) = [1, 2]

      character(len=:), allocatable :: model_dir, model_name
      character(len=128) :: msg1, msg2, msg3, msg4, msg5, msg6
      integer :: ntimes, input_device
      logical :: use_cuda = .false.

      type(torch_module) :: model
      type(torch_tensor), dimension(n_inputs) :: in_tensors
      type(torch_tensor) :: gwfcng_x_tensor, gwfcng_y_tensor

      ! Set flag to .true. via command line argument --explicit_reshape
      ! to explicitly reshape flattened tensors. Default (.false.) is set in setup().
      logical :: explicit_reshape

      ! Set flag to .true. via command line argument --alloc_in_loop
      ! to allocate/deallocate flattened arrays during each loop. Default (.false.) is set in setup().
      ! Only used if explicit_reshape is .true.
      logical :: alloc_in_loop

      print *, "====== DIRECT COUPLED ======"

      call setup(model_dir, model_name, ntimes, n, alloc_in_loop, explicit_reshape, use_cuda)
      if (ntimes .lt. 2) then
        write(*, *) "Error: ntimes must be at least 2"
        return
      end if

      if (use_cuda) then
        input_device = torch_kCUDA
      else
        input_device = torch_kCPU
      end if

      ! Allocate arrays shared with FTorch implementation and read in data
      call init_common_arrays(ntimes, I_MAX, J_MAX, K_MAX, uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, &
                              loop_durations, allocation_durations, deallocation_durations, tensor_creation_durations, &
                              tensor_deletion_durations, inference_durations, all_durations, messages, &
                              start_loop_time, end_loop_time, start_time, end_time)

      ! Allocate arrays and flatten inputs and outputs if --explicit_reshape is set, but --alloc_in_loop is not
      ! if --explicit_reshape and --alloc_in_loop are both set, this is done within each loop instead
      if (.not. alloc_in_loop .and. explicit_reshape) then
        call init_reshaped_arrays(I_MAX, J_MAX, K_MAX, uuu, vvv, lat, psfc, uuu_flattened, vvv_flattened, &
                            lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
      end if

      ! Load model (creation/deletion timed at end)
      model = torch_module_load(model_dir//"/"//model_name)

      do i = 1, ntimes

        ! ------------------------------ Start loop timer ----------------------------
        start_loop_time = omp_get_wtime()

        ! ------------------------------ Start allocation timer ----------------------------
        start_time = omp_get_wtime()
        ! Allocate arrays for flattened inputs and outputs if --alloc_in_loop and --explicit_reshape are set
        if (alloc_in_loop .and. explicit_reshape) then
          call init_reshaped_arrays(I_MAX, J_MAX, K_MAX, uuu, vvv, lat, psfc, uuu_flattened, vvv_flattened, &
          lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
        end if
        end_time = omp_get_wtime()
        allocation_durations(i) = end_time - start_time
        ! ------------------------------ End allocation timer ----------------------------

        ! Create input and output tensors for the model.
        ! ------------------------------ Start tensor creation timer ------------------------------
        start_time = omp_get_wtime()
        if (explicit_reshape) then
          in_tensors(3) = torch_tensor_from_array(lat_reshaped, stride_1D, input_device)
          in_tensors(2) = torch_tensor_from_array(psfc_reshaped, stride_1D, input_device)
        else
          in_tensors(3) = torch_tensor_from_blob(c_loc(lat), int(dims_1D, c_int), int(shape_1D, c_int64_t), int(stride_1D, c_int), torch_wp, input_device)
          in_tensors(2) = torch_tensor_from_blob(c_loc(psfc), int(dims_1D, c_int), int(shape_1D, c_int64_t), int(stride_1D, c_int), torch_wp, input_device)
        end if

        ! Zonal
        if (explicit_reshape) then
          in_tensors(1) = torch_tensor_from_array(uuu_flattened, stride_2D, input_device)
          gwfcng_x_tensor = torch_tensor_from_array(gwfcng_x_flattened, stride_out, torch_kCPU)
        else
          in_tensors(1) = torch_tensor_from_blob(c_loc(uuu), int(dims_2D, c_int), int(shape_2D, c_int64_t), int(stride_2D, c_int), torch_wp, input_device)
          gwfcng_x_tensor = torch_tensor_from_blob(c_loc(gwfcng_x), int(dims_out, c_int), int(shape_out, c_int64_t), int(stride_out, c_int), torch_wp, torch_kCPU)
        end if
        end_time = omp_get_wtime()
        tensor_creation_durations(i) = end_time - start_time
        ! ------------------------------ End tensor creation timer ------------------------------

        ! Run model and Infer
        ! ------------------------------ Start inference timer ------------------------------
        start_time = omp_get_wtime()
        call torch_module_forward(model, in_tensors, n_inputs, gwfcng_x_tensor)
        end_time = omp_get_wtime()
        inference_durations(i) = end_time - start_time
        ! ------------------------------ End inference timer ------------------------------

        ! Clean up here before this points to a new tensor.
        call torch_tensor_delete(in_tensors(1))

        ! Meridional
        ! ------------------------------ Start tensor creation timer ------------------------------
        start_time = omp_get_wtime()
        if (explicit_reshape) then
          in_tensors(1) = torch_tensor_from_array(vvv_flattened, stride_2D, input_device)
          gwfcng_y_tensor = torch_tensor_from_array(gwfcng_y_flattened, stride_out, torch_kCPU)
        else
          in_tensors(1) = torch_tensor_from_blob(c_loc(vvv), int(dims_2D, c_int), int(shape_2D, c_int64_t), int(stride_2D, c_int), torch_wp, input_device)
          gwfcng_y_tensor = torch_tensor_from_blob(c_loc(gwfcng_y), int(dims_out, c_int), int(shape_out, c_int64_t), int(stride_out, c_int), torch_wp, torch_kCPU)
        end if
        end_time = omp_get_wtime()
        tensor_creation_durations(i) = tensor_creation_durations(i) + (end_time - start_time)
        ! ------------------------------ End tensor creation timer ------------------------------

        ! Run model and Infer
        ! ------------------------------ Start inference timer ------------------------------
        start_time = omp_get_wtime()
        call torch_module_forward(model, in_tensors, n_inputs, gwfcng_y_tensor)
        end_time = omp_get_wtime()
        inference_durations(i) = inference_durations(i) + (end_time - start_time)
        ! ------------------------------ End inference timer ------------------------------

        ! ------------------------------ Start inference timer ------------------------------
        ! Include with inference, as necessary for useful output
        start_time = omp_get_wtime()
        if (explicit_reshape) then
          ! Reshape, and assign to gwfcng
          do j = 1, J_MAX
            gwfcng_x(:, j, :) = gwfcng_x_flattened((j - 1) * I_MAX + 1:j * I_MAX, :)
            gwfcng_y(:, j, :) = gwfcng_y_flattened((j - 1) * I_MAX + 1:j * I_MAX, :)
          end do
        end if
        end_time = omp_get_wtime()
        inference_durations(i) = inference_durations(i) + (end_time - start_time)
        ! ------------------------------ End inference timer ------------------------------

        ! Clean up.
        ! ------------------------------ Start tensor deletion timer ------------------------------
        start_time = omp_get_wtime()
        call torch_tensor_delete(gwfcng_y_tensor)
        call torch_tensor_delete(gwfcng_x_tensor)
        do ii = 1, n_inputs
          call torch_tensor_delete(in_tensors(ii))
        end do
        end_time = omp_get_wtime()
        tensor_deletion_durations(i) = end_time - start_time
        ! ------------------------------ End tensor deletion timer ------------------------------

        ! Check error
        call assert(gwfcng_x, gwfcng_x_ref, "Check x", rtol_opt=1.0e-7_wp)
        call assert(gwfcng_y, gwfcng_y_ref, "Check y", rtol_opt=1.0e-7_wp)

        ! ------------------------------ Start deallocation timer ------------------------------
        start_time = omp_get_wtime()
        ! Deallocate arrays for flattened inputs and outputs if --alloc_in_loop and --explicit_reshape are set
        if (alloc_in_loop .and. explicit_reshape) then
          call deallocate_reshaped_arrays(uuu_flattened, vvv_flattened, lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
        end if
        end_time = omp_get_wtime()
        deallocation_durations(i) = end_time - start_time
        ! ------------------------------ End deallocation timer -----------------------------

        end_loop_time = omp_get_wtime()
        loop_durations(i) = end_loop_time - start_loop_time
        ! ------------------------------ End loop timer ----------------------------

        write(msg1, '(A, I18, A, F10.6, A)') "check iteration inference", i, " (", inference_durations(i), " s)"
        write(msg2, '(A, I13, A, F10.6, A)') "check iteration create tensors", i, " (", tensor_creation_durations(i), " s)"
        write(msg3, '(A, I13, A, F10.6, A)') "check iteration delete tensors", i, " (", tensor_deletion_durations(i), " s)"
        write(msg4, '(A, I12, A, F10.6, A)') "check iteration allocate arrays", i, " (", allocation_durations(i), " s)"
        write(msg5, '(A, I10, A, F10.6, A)') "check iteration deallocate arrays", i, " (", deallocation_durations(i), " s)"
        write(msg6, '(A, I18, A, F11.6, A)') "check iteration full loop", i, " (", loop_durations(i), " s)"
        print *, trim(msg1)
        print *, trim(msg2)
        print *, trim(msg3)
        print *, trim(msg4)
        print *, trim(msg5)
        print *, trim(msg6)

      end do

      ! Call individual print for loop, to avoid adding to combined mean
      call print_time_stats(loop_durations, "full loop")

      all_durations(:, 1) = allocation_durations
      all_durations(:, 2) = deallocation_durations
      all_durations(:, 3) = tensor_creation_durations
      all_durations(:, 4) = tensor_deletion_durations
      all_durations(:, 5) = inference_durations
      messages = [character(len=20) :: "array allocation", "array deallocation", &
                  "tensor creation", "tensor deletion", "forward pass"]
      call print_all_time_stats(all_durations, messages)

      call deallocate_common_arrays(uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, loop_durations, &
                                    allocation_durations, deallocation_durations, tensor_creation_durations, tensor_deletion_durations, &
                                    inference_durations, all_durations, messages)

      ! Deallocate arrays for flattened inputs and outputs if --explicit_reshape is set, but --alloc_in_loop is not
      ! if --explicit_reshape and --alloc_in_loop are both set, this is done within each loop instead
      if (.not. alloc_in_loop .and. explicit_reshape) then
        call deallocate_reshaped_arrays(uuu_flattened, vvv_flattened, lat_reshaped, psfc_reshaped, gwfcng_x_flattened, gwfcng_y_flattened)
      end if

    end subroutine main

    subroutine init_common_arrays(ntimes, I_MAX, J_MAX, K_MAX, uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, &
                                  loop_durations, allocation_durations, deallocation_durations, tensor_creation_durations, &
                                  tensor_deletion_durations, inference_durations, all_durations, messages, &
                                  start_loop_time, end_loop_time, start_time, end_time)

      implicit none

      integer, intent(in):: ntimes, I_MAX, J_MAX, K_MAX

      real(wp), intent(out), dimension(:,:,:), allocatable :: uuu, vvv, gwfcng_x, gwfcng_y
      real(wp), intent(out), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
      real(wp), intent(out), dimension(:,:), allocatable :: lat, psfc

      real(dp), intent(out), dimension(:), allocatable :: loop_durations, inference_durations, allocation_durations
      real(dp), intent(out), dimension(:), allocatable :: deallocation_durations, tensor_creation_durations, tensor_deletion_durations
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
      allocate(loop_durations(ntimes))
      allocate(allocation_durations(ntimes))
      allocate(deallocation_durations(ntimes))
      allocate(tensor_creation_durations(ntimes))
      allocate(tensor_deletion_durations(ntimes))
      allocate(inference_durations(ntimes))
      allocate(all_durations(ntimes, 5))
      allocate(messages(5))

      ! Initialise timings with arbitrary large values
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

    subroutine deallocate_common_arrays(uuu, vvv, gwfcng_x, gwfcng_y, gwfcng_x_ref, gwfcng_y_ref, lat, psfc, loop_durations, &
                                        allocation_durations, deallocation_durations, tensor_creation_durations, &
                                        tensor_deletion_durations, inference_durations, all_durations, messages)

      implicit none

      real(dp), intent(inout), dimension(:), allocatable :: loop_durations, inference_durations, allocation_durations
      real(dp), intent(inout), dimension(:), allocatable :: deallocation_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), intent(inout), dimension(:,:), allocatable :: all_durations
      character(len=20), intent(inout), dimension(:), allocatable :: messages

      real(wp), intent(inout), dimension(:,:,:), allocatable :: uuu, vvv, gwfcng_x, gwfcng_y
      real(wp), intent(inout), dimension(:,:,:), allocatable :: gwfcng_x_ref, gwfcng_y_ref
      real(wp), intent(inout), dimension(:,:), allocatable :: lat, psfc

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
