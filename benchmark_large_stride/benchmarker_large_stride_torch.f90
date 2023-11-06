program benchmark_stride_test

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, print_time_stats, print_all_time_stats
  use :: ftorch
  use :: precision, only: wp, dp

  implicit none

  integer, parameter :: torch_wp = torch_kFloat32

  call main()

  contains

    subroutine main()

      implicit none

      integer :: i, ii, n
      real(wp), dimension(:,:), allocatable, target :: big_array, big_result

      real(dp) :: start_time, end_time, start_loop_time, end_loop_time
      real(dp), dimension(:), allocatable :: module_load_durations, module_delete_durations, loop_durations
      real(dp), dimension(:), allocatable :: inference_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), dimension(:,:), allocatable :: all_durations
      character(len=20), dimension(:), allocatable :: messages

      integer(c_int), parameter :: n_inputs = 1
      integer(c_int64_t) :: shape_2d(2)
      integer(c_int) :: stride_2d(2)

      character(len=:), allocatable :: model_dir, model_name
      character(len=128) :: msg1, msg2, msg3, msg4
      integer :: ntimes

      type(torch_tensor) :: result_tensor
      type(torch_tensor), dimension(n_inputs), target :: input_array
      type(torch_module) :: model

      print *, "====== DIRECT COUPLED ======"

      call setup(model_dir, model_name, ntimes, n)

      allocate(big_array(n, n))
      allocate(big_result(n, n))
      allocate(module_load_durations(ntimes))
      allocate(module_delete_durations(ntimes))
      allocate(loop_durations(ntimes))
      allocate(tensor_creation_durations(ntimes))
      allocate(tensor_deletion_durations(ntimes))
      allocate(inference_durations(ntimes))
      allocate(all_durations(ntimes, 5))
      allocate(messages(5))

      ! Initialise timings with arbitrary large values
      module_load_durations(:) = 100.
      module_delete_durations(:) = 100.
      loop_durations(:) = 100.
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
      model = torch_module_load(model_dir//"/"//model_name)

      shape_2d = (/ n, n /)
      stride_2d = (/ 1, 2 /)

      do i = 1, ntimes

        ! ------------------------------ Start loop timer ------------------------------
        start_loop_time = omp_get_wtime()

        call random_number(big_array)

        ! Create input and output tensors for the model.
        ! ------------------------------ Start tensor creation timer ------------------------------
        start_time = omp_get_wtime()
        input_array(1) = torch_tensor_from_blob(c_loc(big_array), 2, shape_2d, torch_wp, torch_kCPU, stride_2d)
        result_tensor = torch_tensor_from_blob(c_loc(big_result), 2, shape_2d, torch_wp, torch_kCPU, stride_2d)
        end_time = omp_get_wtime()
        tensor_creation_durations(i) = end_time - start_time
        ! ------------------------------ End tensor creation timer ------------------------------

        ! ------------------------------ Start inference timer ------------------------------
        start_time = omp_get_wtime()
        call torch_module_forward(model, input_array, n_inputs, result_tensor)
        end_time = omp_get_wtime()
        inference_durations(i) = end_time - start_time
        ! ------------------------------ End inference timer -------------------------------

        ! Clean up.
        ! ------------------------------ Start tensor deletion timer ------------------------------
        start_time = omp_get_wtime()
        call torch_tensor_delete(result_tensor)
        do ii = 1, n_inputs
          call torch_tensor_delete(input_array(ii))
        end do
        end_time = omp_get_wtime()
        tensor_deletion_durations(i) = end_time - start_time
        ! ------------------------------ End tensor deletion timer ------------------------------

        end_loop_time = omp_get_wtime()
        loop_durations(i) = end_loop_time - start_loop_time
        ! ------------------------------ End loop timer ----------------------------

        ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
        big_array(1, 2) = -1.0*big_array(1, 2)
        call assert(big_array, big_result/2., test_name="Check array")

        write(msg1, '(A, I10, A, F10.6, A)') "check iteration create tensors", i, " (", tensor_creation_durations(i), " s)"
        write(msg2, '(A, I15, A, F10.6, A)') "check iteration inference", i, " (", inference_durations(i), " s)"
        write(msg3, '(A, I10, A, F10.6, A)') "check iteration delete tensors", i, " (", tensor_deletion_durations(i), " s)"
        write(msg4, '(A, I18, A, F11.6, A)') "check iteration full loop", i, " (", loop_durations(i), " s)"
        print *, trim(msg1)
        print *, trim(msg2)
        print *, trim(msg3)
        print *, trim(msg4)

      end do

      ! Delete model (creation/deletion timed at end)
      call torch_module_delete(model)

      call time_module(ntimes, model_dir, model_name, module_load_durations, module_delete_durations)

      ! Call individual print for loop, to avoid adding to combined mean
      call print_time_stats(loop_durations, "full loop")

      all_durations(:, 1) = module_load_durations
      all_durations(:, 2) = module_delete_durations
      all_durations(:, 3) = tensor_creation_durations
      all_durations(:, 4) = tensor_deletion_durations
      all_durations(:, 5) = inference_durations
      messages = [character(len=20) :: "module creation", "module deletion", "tensor creation", "tensor deletion", "forward pass"]
      call print_all_time_stats(all_durations, messages)

      deallocate(big_array)
      deallocate(big_result)
      deallocate(module_load_durations)
      deallocate(module_delete_durations)
      deallocate(loop_durations)
      deallocate(tensor_creation_durations)
      deallocate(tensor_deletion_durations)
      deallocate(inference_durations)
      deallocate(all_durations)
      deallocate(messages)

    end subroutine main

    subroutine time_module(ntimes, model_dir, model_name, module_load_durations, module_delete_durations)

      implicit none

      integer, intent(in) :: ntimes
      real(dp), dimension(:), intent(out) :: module_load_durations, module_delete_durations
      integer :: i
      real(dp) :: start_time, end_time
      character(len=*), intent(in) :: model_dir, model_name
      type(torch_module) :: model

      do i = 1, ntimes
        ! ------------------------------ Start module load timer ------------------------------
        start_time = omp_get_wtime()
        model = torch_module_load(model_dir//"/"//model_name)
        end_time = omp_get_wtime()
        module_load_durations(i) = end_time - start_time
        ! ------------------------------ End module load timer ------------------------------

        ! ------------------------------ Start module deletion timer ------------------------------
        start_time = omp_get_wtime()
        call torch_module_delete(model)
        end_time = omp_get_wtime()
        module_delete_durations(i) = end_time - start_time
        ! ------------------------------ End module deletion timer ------------------------------
      end do

    end subroutine time_module

end program
