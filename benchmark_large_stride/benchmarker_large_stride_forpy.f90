program benchmark_stride_test

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
      real(wp), dimension(:,:), allocatable, asynchronous :: big_array, big_result

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

      type(ndarray) :: big_result_nd, big_array_nd

      print *, "====== FORPY ======"

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
#ifdef USETS
      print *, "load torchscript model"
#else
      print *, "generate model in python runtime"
#endif
      call load_module(model_dir, model_name, run_emulator, model)

      do i = 1, ntimes

        ! ------------------------------ Start loop timer ------------------------------
        start_loop_time = omp_get_wtime()

        call random_number(big_array)

        ! creates numpy arrays
        ! ------------------------------ Start tensor creation timer ------------------------------
        start_time = omp_get_wtime()
        ie = ndarray_create_nocopy(big_array_nd, big_array)
        ie = ndarray_create_nocopy(big_result_nd, big_result)

        ! create model input args as tuple
        ie = tuple_create(args,3)
        ie = args%setitem(0, model)
        ie = args%setitem(1, big_array_nd)
        ie = args%setitem(2, big_result_nd)
        end_time = omp_get_wtime()
        tensor_creation_durations(i) = end_time - start_time
        ! ------------------------------ End tensor creation timer ------------------------------

        ! ------------------------------ Start inference timer ------------------------------
        start_time = omp_get_wtime()
        ie = call_py_noret(run_emulator, "compute", args)
        end_time = omp_get_wtime()
        inference_durations(i) = end_time - start_time
        ! ------------------------------ End inference timer --------------------------------

        if (ie .ne. 0) then
          call err_print
          call error_mesg(__FILE__, __LINE__, "inference call failed")
        end if

        ! Clean up.
        ! ------------------------------ Start tensor deletion timer ------------------------------
        start_time = omp_get_wtime()
        call big_result_nd%destroy
        call big_array_nd%destroy
        call args%destroy
        end_time = omp_get_wtime()
        tensor_deletion_durations(i) = end_time - start_time
        ! ------------------------------ End tensor deletion timer ------------------------------

        end_loop_time = omp_get_wtime()
        loop_durations(i) = end_loop_time - start_loop_time
        ! ------------------------------ End loop timer ----------------------------


        ! the forward model is deliberately non-symmetric to check for difference in Fortran and C--type arrays.
        big_array(1, 2) = -1.0*big_array(1, 2)
        call assert(big_array, big_result/2., test_name="Check array")

        write(msg1, '(A, I10, A, F10.3, A)') "check iteration create tensors", i, " (", tensor_creation_durations(i), " s)"
        write(msg2, '(A, I15, A, F10.3, A)') "check iteration inference", i, " (", inference_durations(i), " s)"
        write(msg3, '(A, I10, A, F10.3, A)') "check iteration delete tensors", i, " (", tensor_deletion_durations(i), " s)"
        write(msg4, '(A, I18, A, F11.4, A)') "check iteration full loop", i, " (", loop_durations(i), " s)"
        print *, trim(msg1)
        print *, trim(msg2)
        print *, trim(msg3)
        print *, trim(msg4)

      end do

      call time_module(ntimes, model_dir, model_name, module_load_durations, module_delete_durations, run_emulator, model)

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
      ie = str_create(filename, trim(model_dir//"/"//"saved_large_stride_model_cpu.pt"))
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

end program
