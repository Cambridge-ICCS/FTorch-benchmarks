program benchmark_resnet

  use, intrinsic :: iso_c_binding
  use :: omp_lib, only : omp_get_wtime
  use :: utils, only : assert, setup, error_mesg, print_time_stats, print_all_time_stats
  use :: forpy_mod, only: import_py, module_py, call_py, object, ndarray, &
                          forpy_initialize, forpy_finalize, tuple, tuple_create, &
                          ndarray_create, err_print, call_py_noret, list, &
                          get_sys_path, ndarray_create_nocopy, str, str_create
  use :: precision, only: wp, dp

  implicit none

  call main()

  contains

    subroutine main()

      implicit none

      integer :: i, n
      real(wp), dimension(:,:,:,:), allocatable, asynchronous :: in_data
      real(wp), dimension(:,:), allocatable, asynchronous :: out_data

      real(dp) :: start_time, end_time, start_loop_time, end_loop_time
      real(dp), dimension(:), allocatable :: module_load_durations, module_delete_durations, loop_durations
      real(dp), dimension(:), allocatable :: inference_durations, tensor_creation_durations, tensor_deletion_durations
      real(dp), dimension(:,:), allocatable :: all_durations
      character(len=20), dimension(:), allocatable :: messages

      integer :: ie
      type(module_py) :: run_emulator
      type(object) :: model
      type(tuple) :: args

      character(len=:), allocatable :: model_dir, model_name
      character(len=128) :: msg1, msg2, msg3, msg4
      integer :: ntimes
      character(len=10) :: input_device
      logical :: use_cuda = .false.

      type(ndarray) :: out_data_nd, in_data_nd

      ! Binary file containing input tensor
      character(len=*), parameter :: data_file = '../resnet_model/image_tensor.dat'

      ! Length of tensor and number of categories
      integer, parameter :: tensor_length = 150528

      ! Outputs
      integer :: idx(2)
      real(wp), dimension(:,:), allocatable :: probabilities
      real(wp), parameter :: expected_prob = 0.8846225142478943
      real(wp) :: probability

      print *, "====== FORPY ======"

      call setup(model_dir, model_name, ntimes, n, use_cuda=use_cuda)

      if (use_cuda) then
        input_device = "cuda"
      else
        input_device = "cpu"
      end if

      allocate(in_data(1, 3, 224, 224))
      allocate(out_data(1, 1000))
      allocate(probabilities(1, 1000))
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
#ifdef USETS
      print *, "load torchscript model"
#else
      print *, "generate model in python runtime"
#endif
      call load_module(model_dir, model_name, run_emulator, model, use_cuda)

      call load_data(data_file, tensor_length, in_data)

      do i = 1, ntimes

        ! ------------------------------ Start loop timer ----------------------------
        start_loop_time = omp_get_wtime()

        ! creates numpy arrays
        ! ------------------------------ Start tensor creation timer ------------------------------
        start_time = omp_get_wtime()
        ie = ndarray_create_nocopy(in_data_nd, in_data)
        ie = ndarray_create_nocopy(out_data_nd, out_data)

        ! create model input args as tuple
        ie = tuple_create(args, 4)
        ie = args%setitem(0, model)
        ie = args%setitem(1, in_data_nd)
        ie = args%setitem(2, trim(input_device))
        ie = args%setitem(3, out_data_nd)
        end_time = omp_get_wtime()
        tensor_creation_durations(i) = end_time - start_time
        ! ------------------------------ End tensor creation timer ------------------------------

        ! ------------------------------ Start inference timer ------------------------------
        start_time = omp_get_wtime()
        ie = call_py_noret(run_emulator, "compute", args)
        end_time = omp_get_wtime()
        inference_durations(i) = end_time - start_time
        ! ------------------------------ End inference timer -----------------------------

        if (ie .ne. 0) then
            call err_print
            call error_mesg(__FILE__, __LINE__, "inference call failed")
        end if

        ! Clean up.
        ! ------------------------------ Start tensor deletion timer ------------------------------
        start_time = omp_get_wtime()
        call out_data_nd%destroy
        call in_data_nd%destroy
        call args%destroy
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
        call assert(probability, expected_prob, test_name="Check probability", rtol_opt=1.0e-5_wp)

        write(msg1, '(A, I10, A, F10.6, A)') "check iteration create tensors", i, " (", tensor_creation_durations(i), " s)"
        write(msg2, '(A, I15, A, F10.6, A)') "check iteration inference", i, " (", inference_durations(i), " s)"
        write(msg3, '(A, I10, A, F10.6, A)') "check iteration delete tensors", i, " (", tensor_deletion_durations(i), " s)"
        write(msg4, '(A, I18, A, F11.6, A)') "check iteration full loop", i, " (", loop_durations(i), " s)"
        print *, trim(msg1)
        print *, trim(msg2)
        print *, trim(msg3)
        print *, trim(msg4)

      end do

      module_load_durations(:) = 0.
      module_delete_durations(:) = 0.
      call forpy_finalize

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
      real(wp), dimension(:,:,:,:), intent(out) :: in_data

      real(wp) :: flat_data(tensor_length)
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

      real(wp), dimension(:,:), intent(in) :: out_data
      real(wp), dimension(:,:), intent(out) :: probabilities
      real(wp) :: prob_sum

      ! Apply softmax function to calculate probabilties
      probabilities = exp(out_data)
      prob_sum = sum(probabilities)
      probabilities = probabilities / prob_sum

    end subroutine calc_probs

    subroutine load_module(model_dir, model_name, run_emulator, model, use_cuda)

      implicit none

      character(len=*), intent(in) :: model_dir, model_name
      logical, intent(in) :: use_cuda
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
      if (use_cuda) then
      ie = str_create(filename, trim(model_dir//"/"//"saved_resnet18_model_gpu.pt"))
      else
        ie = str_create(filename, trim(model_dir//"/"//"saved_resnet18_model_cpu.pt"))
      end if
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

end program benchmark_resnet
