program benchmark_resnet_test

  use, intrinsic :: iso_c_binding, only: c_int64_t, c_null_char, c_loc
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, print_time_stats
  ! Import our library for interfacing with PyTorch
  use :: ftorch
  use :: precision, only: c_sp, c_dp, sp, dp

  implicit none

  ! Define working precision for C primitives
  ! Precision must match `wp` in resnet18.py and `wp_torch` in pt2ts.py
  integer, parameter :: c_wp = c_sp
  integer, parameter :: wp = sp
  integer, parameter :: torch_wp = torch_kFloat32

  call main()

  contains

    subroutine main()

    integer :: i, ii, n
    double precision :: start_time, end_time
    double precision, allocatable :: durations(:)

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
    character(len=128) :: msg
    integer :: ntimes, input_device
    logical :: use_cuda = .false.

    type(torch_module) :: model
    type(torch_tensor), dimension(1) :: in_tensor
    type(torch_tensor) :: out_tensor

    ! Binary file containing input tensor
    character(len=*), parameter :: filename = '../resnetmodel/image_tensor.dat'

    ! Length of tensor and number of categories
    integer, parameter :: tensor_length = 150528

    ! Outputs
    integer :: idx(2)
    real(wp), dimension(:,:), allocatable :: probabilities
    real(wp), parameter :: expected_prob = 0.8846225142478943
    real(wp) :: probability

    print *, "====== DIRECT COUPLED ======"

    call setup(model_dir, model_name, ntimes, n, use_cuda)

    allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
    allocate(out_data(out_shape(1), out_shape(2)))
    allocate(durations(ntimes))
    allocate(probabilities(out_shape(1), out_shape(2)))

    model = torch_module_load(model_dir//"/"//model_name//C_NULL_CHAR)

    ! Initialise data - previously in loop, but not modified?
    call load_data(filename, tensor_length, in_data)

    if (use_cuda) then
        input_device = torch_kCUDA
      else
        input_device = torch_kCPU
    end if

    do i = 1, ntimes

      start_time = omp_get_wtime()

      ! Create input and output tensors for the model.
      in_tensor(1) = torch_tensor_from_blob(c_loc(in_data), in_dims, in_shape, torch_kFloat32, input_device, in_layout)
      out_tensor = torch_tensor_from_blob(c_loc(out_data), out_dims, out_shape, torch_kFloat32, torch_kCPU, out_layout)

      call torch_module_forward(model, in_tensor, n_inputs, out_tensor)

      ! Clean up.
      call torch_tensor_delete(out_tensor)
      do ii = 1, n_inputs
        call torch_tensor_delete(in_tensor(ii))
      end do

      end_time = omp_get_wtime()
      durations(i) = end_time-start_time

      ! Calculate probabilities and output results
      call calc_probs(out_data, probabilities)
      idx = maxloc(probabilities)
      probability = maxval(probabilities)

      ! Check top probability matches expected value
      call assert(probability, expected_prob, test_name="Check probability", rtol_opt=1e-2)

      ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
      write(msg, '(A, I8, A, F10.3, A)') "check iteration ", i, " (", durations(i), " s) [omp]"
      print *, trim(msg)
    end do

    call print_time_stats(durations)


    call torch_module_delete(model)

    deallocate(in_data)
    deallocate(out_data)
    deallocate(durations)
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

end program benchmark_resnet_test
