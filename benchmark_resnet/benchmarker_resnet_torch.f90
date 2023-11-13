program benchmark_resnet_test

  use, intrinsic :: iso_c_binding, only: c_int64_t, c_loc
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, print_time_stats, print_all_time_stats
  ! Import our library for interfacing with PyTorch
  use :: ftorch
  ! Define working precision for C primitives and Fortran reals
  ! Precision must match `wp` in resnet18.py and `wp_torch` in pt2ts.py
  use :: precision, only: c_wp, wp, dp

  implicit none

  integer, parameter :: torch_wp = torch_kFloat32

  call main()

  contains

    subroutine main()

      implicit none

      integer :: i, ii, n
      real(dp) :: start_time, end_time, start_loop_time, end_loop_time
      real(dp), dimension(:), allocatable :: module_load_durations, module_delete_durations, loop_durations
      real(dp), dimension(:), allocatable :: inference_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), dimension(:,:), allocatable :: all_durations
      character(len=20), dimension(:), allocatable :: messages

      real(c_wp), dimension(:,:,:,:), allocatable, target :: in_data
      integer(c_int), parameter :: n_inputs = 1
      real(c_wp), dimension(:,:), allocatable, target :: out_data

      integer(c_int), parameter :: in_dims = 4
      integer(c_int64_t) :: in_shape(in_dims) = [1, 3, 224, 224]
      integer(c_int) :: in_layout(in_dims) = [1,2,3,4]
      integer(c_int), parameter :: out_dims = 2
      integer(c_int64_t) :: out_shape(out_dims) = [1, 1000]
      integer(c_int) :: out_layout(out_dims) = [1,2]

      character(len=:), allocatable :: model_dir, model_name
      character(len=128) :: msg1, msg2, msg3, msg4
      integer :: ntimes, input_device
      logical :: use_cuda = .false.

      type(torch_module) :: model
      type(torch_tensor), dimension(1) :: in_tensor
      type(torch_tensor) :: out_tensor

      ! Binary file containing input tensor
      character(len=*), parameter :: filename = '../resnet_model/image_tensor.dat'

      ! Length of tensor and number of categories
      integer, parameter :: tensor_length = 150528

      ! Outputs
      integer :: idx(2)
      real(wp), dimension(:,:), allocatable :: probabilities
      real(wp), parameter :: expected_prob = 0.8846225142478943
      real(wp) :: probability

      print *, "====== DIRECT COUPLED ======"

      call setup(model_dir, model_name, ntimes, n, use_cuda=use_cuda)

      if (use_cuda) then
        input_device = torch_kCUDA
      else
        input_device = torch_kCPU
      end if

      allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
      allocate(out_data(out_shape(1), out_shape(2)))
      allocate(probabilities(out_shape(1), out_shape(2)))

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

      ! Initialise data - previously in loop, but not modified?
      call load_data(filename, tensor_length, in_data)

      do i = 1, ntimes

        ! ------------------------------ Start loop timer ----------------------------
        start_loop_time = omp_get_wtime()

        ! Create input and output tensors for the model.
        ! ------------------------------ Start tensor creation timer ------------------------------
        start_time = omp_get_wtime()
        in_tensor(1) = torch_tensor_from_blob(c_loc(in_data), in_dims, in_shape, torch_wp, input_device, in_layout)
        out_tensor = torch_tensor_from_blob(c_loc(out_data), out_dims, out_shape, torch_wp, torch_kCPU, out_layout)
        end_time = omp_get_wtime()
        tensor_creation_durations(i) = end_time - start_time
        ! ------------------------------ End tensor creation timer ------------------------------

        ! ------------------------------ Start inference timer ------------------------------
        start_time = omp_get_wtime()
        call torch_module_forward(model, in_tensor, n_inputs, out_tensor)
        end_time = omp_get_wtime()
        inference_durations(i) = end_time - start_time
        ! ------------------------------ End inference timer -------------------------------

        ! Clean up.
        ! ------------------------------ Start tensor deletion timer ------------------------------
        start_time = omp_get_wtime()
        call torch_tensor_delete(out_tensor)
        do ii = 1, n_inputs
          call torch_tensor_delete(in_tensor(ii))
        end do
        end_time = omp_get_wtime()
        tensor_deletion_durations(i) = end_time - start_time
        ! ------------------------------ End tensor deletion timer ------------------------------

        end_loop_time = omp_get_wtime()
        loop_durations(i) = end_loop_time - start_loop_time
        ! ------------------------------ End loop timer ----------------------------

        ! Calculate probabilities and output results
        call calc_probs(out_data, probabilities)
        idx = maxloc(probabilities)
        probability = maxval(probabilities)

        ! Check top probability matches expected value
        call assert(probability, expected_prob, test_name="Check probability", rtol_opt=1.0e-2_wp)

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

      deallocate(in_data)
      deallocate(out_data)
      deallocate(module_load_durations)
      deallocate(module_delete_durations)
      deallocate(loop_durations)
      deallocate(tensor_creation_durations)
      deallocate(tensor_deletion_durations)
      deallocate(inference_durations)
      deallocate(all_durations)
      deallocate(messages)
      deallocate(probabilities)

    end subroutine main

    subroutine load_data(filename, tensor_length, in_data)

      implicit none

      character(len=*), intent(in) :: filename
      integer, intent(in) :: tensor_length
      real(c_wp), dimension(:,:,:,:), intent(out) :: in_data

      real(c_wp) :: flat_data(tensor_length)
      integer :: ios
      character(len=100) :: ioerrmsg

      ! Read input tensor from Python script
      open(unit=10, file=filename, status='old', access='stream', form='unformatted', action="read", iostat=ios, iomsg=ioerrmsg)
      if (ios /= 0) then
      print *, ioerrmsg
      stop 1
      end if

      read(10, iostat=ios, iomsg=ioerrmsg) flat_data
      if (ios /= 0) then
          print *, ioerrmsg
          stop 1
      end if

      close(10)

      ! Reshape data to tensor input shape
      ! This assumes the data from Python was transposed before saving
      in_data = reshape(flat_data, shape(in_data))

    end subroutine load_data

    subroutine calc_probs(out_data, probabilities)

      implicit none

      real(c_wp), dimension(:,:), intent(in) :: out_data
      real(wp), dimension(:,:), intent(out) :: probabilities
      real(wp) :: prob_sum

      ! Apply softmax function to calculate probabilties
      probabilities = exp(out_data)
      prob_sum = sum(probabilities)
      probabilities = probabilities / prob_sum

    end subroutine calc_probs

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

end program benchmark_resnet_test
