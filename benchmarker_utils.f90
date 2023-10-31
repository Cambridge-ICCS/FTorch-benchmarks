module utils

  use :: precision, only: wp, dp

  implicit none

  interface print_time_stats
    module procedure print_time_stats_dp
  end interface

  interface print_all_time_stats
    module procedure print_all_time_stats_dp
  end interface

  interface assert
    module procedure assert_real, assert_real_2d, assert_real_3d_dp
  end interface

  interface print_assert
    module procedure print_assert_real, print_assert_dp
  end interface

  contains

  subroutine print_assert_real(test_name, is_close, relative_error)

    implicit none

    character(len=*), intent(in) :: test_name
    logical, intent(in) :: is_close
    real(wp), intent(in) :: relative_error
    character(len=15) :: pass, fail

    fail = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
    pass = char(27)//'[32m'//'PASSED'//char(27)//'[0m'

    if (is_close) then
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') pass, trim(test_name), relative_error
    else
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') fail, trim(test_name), relative_error
    end if

  end subroutine print_assert_real

  subroutine print_assert_dp(test_name, is_close, relative_error)

    implicit none

    character(len=*), intent(in) :: test_name
    logical, intent(in) :: is_close
    real(dp), intent(in) :: relative_error
    character(len=15) :: pass, fail

    fail = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
    pass = char(27)//'[32m'//'PASSED'//char(27)//'[0m'

    if (is_close) then
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') pass, trim(test_name), relative_error
    else
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') fail, trim(test_name), relative_error
    end if

  end subroutine print_assert_dp

  subroutine assert_real(a, b, test_name, rtol_opt)

    implicit none

    character(len=*), intent(in) :: test_name
    real(wp), intent(in) :: a, b
    real(wp), intent(in), optional :: rtol_opt
    real(wp) :: relative_error, rtol

    if (.not. present(rtol_opt)) then
      rtol = 1.0e-5_wp
    else
      rtol = rtol_opt
    end if

    relative_error = abs(a/b - 1.0_wp)
    call print_assert(test_name, (rtol > relative_error), relative_error)

  end subroutine assert_real

  subroutine assert_real_2d(a, b, test_name, rtol_opt)

    implicit none

    character(len=*), intent(in) :: test_name
    real(wp), intent(in), dimension(:,:) :: a, b
    real(wp), intent(in), optional :: rtol_opt
    real(wp) :: relative_error, rtol

    if (.not. present(rtol_opt)) then
      rtol = 1.0e-5_wp
    else
      rtol = rtol_opt
    end if

    relative_error = maxval(abs(a/b - 1.0_wp))
    call print_assert(test_name, (rtol > relative_error), relative_error)

  end subroutine assert_real_2d

  subroutine assert_real_3d_dp(a, b, test_name, rtol_opt)

    implicit none

    character(len=*), intent(in) :: test_name
    real(dp), intent(in), dimension(:,:,:) :: a, b
    real(dp), intent(in), optional :: rtol_opt
    real(dp) :: relative_error, rtol

    if (.not. present(rtol_opt)) then
      rtol = 1.0e-5_dp
    else
      rtol = rtol_opt
    end if

    relative_error = maxval(abs(a/b - 1.0_dp))
    call print_assert(test_name, (rtol > relative_error), relative_error)

  end subroutine assert_real_3d_dp

  subroutine print_all_time_stats_dp(durations, messages)

    implicit none

    real(dp), intent(in) :: durations(:,:)
    character(len=*), optional, intent(in) :: messages(:)
    integer :: n, i
    real(dp), allocatable :: means(:)
    real(dp) :: mean

    allocate(means(size(durations, 2)))
    n = size(durations(2:, 1), 1)

    do i = 1, size(durations, 2)
      call print_time_stats_dp(durations(:, i), messages(i))
      means(i) = sum(durations(2:, i)) / n
    end do

    mean = sum(means)

    write (*,*) "----------- Combined results ----------------"
    write(*,'(A,F18.4,A)') "Overall mean (s): ", mean
    write(*,'(A,I22)')     "Sample size: ", n
    write (*,*) "---------------------------------------------"

    deallocate(means)

  end subroutine print_all_time_stats_dp

  subroutine print_time_stats_dp(durations, message)

    implicit none

    real(dp), intent(in) :: durations(:)
    character(len=*), optional, intent(in) :: message
    real(dp) :: mean, var, stddev
    integer :: i, n

    ! skip the first element because this is always slower
    n = size(durations(2:), 1)

    mean = sum(durations(2:)) / n

    var = 0.0_dp

    do i = 2, n
      var = var + ( (durations(i) - mean)**2.0_dp / (n - 1.0_dp) ) ! (n - 1) here is for corrected sample standard deviation
    end do

    stddev = sqrt(var)

    if (present(message)) then
      write(*,*) trim(adjustl(message))
    endif
    write(*,'(A,F10.4,A)') "min    time taken (s): ", minval(durations(2:))
    write(*,'(A,F10.4,A)') "max    time taken (s): ", maxval(durations(2:))
    write(*,'(A,F10.4,A)') "mean   time taken (s): ", mean
    write(*,'(A,F10.4,A)') "stddev time taken (s): ", stddev
    write(*,'(A,I10)')     "sample size          : ", n

  end subroutine print_time_stats_dp

  subroutine setup(model_dir, model_name, ntimes, n, alloc_in_loop, explicit_reshape)

    implicit none

    character(len=:), allocatable, intent(inout) :: model_dir, model_name
    integer, intent(out) :: ntimes, n
    logical, intent(out), optional :: alloc_in_loop, explicit_reshape

    character(len=1024) :: model_dir_temp, model_name_temp
    character(len=16) :: ntimes_char, n_char
    character(len=32) :: flag
    integer :: i

    ! Parse required arguments
    if (command_argument_count() .lt. 4 .or. command_argument_count() .gt. 6) then
      call error_mesg(__FILE__, __LINE__, "Usage: benchmarker <model-dir> <model-name> <ntimes> <N> <--alloc_in_loop[optional]> <--explicit_reshape[optional]>")
    endif

    call get_command_argument(1, model_dir_temp)
    call get_command_argument(2, model_name_temp)
    call get_command_argument(3, ntimes_char)
    call get_command_argument(4, n_char)

    read(ntimes_char, *) ntimes
    read(n_char, *) n
    model_dir = trim(adjustl(model_dir_temp))
    model_name = trim(adjustl(model_name_temp))

    write(*,'("Running model: ", A, "/", A, " ", I0, " times.")') model_dir, model_name, ntimes

    if (present(alloc_in_loop)) then
      alloc_in_loop = .false.
    end if
    if (present(explicit_reshape)) then
      explicit_reshape = .false.
    end if

    if (command_argument_count() .gt. 4) then
      write(*,*) "Optional settings:"
      do i = 5, command_argument_count()
        call get_command_argument(i, flag)

        select case (flag)

          ! If .true., explicitly reshape FTorch arrays, rather than implicitly
          case ('--explicit_reshape')
            if (present(explicit_reshape)) then
              explicit_reshape = .true.
            else
              print '(2a, /)', 'explicit_reshape must be passed to setup() to use ', flag
              stop
            end if

          ! If .true., allocate/deallocate arrays for each loop
          case ('--alloc_in_loop')
            if (present(alloc_in_loop)) then
              alloc_in_loop = .true.
            else
              print '(2a, /)', 'alloc_in_loop must be passed to setup() to use ', flag
              stop
            end if

          case default
            print '(2a, /)', 'Error: unrecognised command-line option: ', flag
            stop
        end select
      end do
    end if

    if (present(explicit_reshape) .and. present(alloc_in_loop)) then
      if (.not. explicit_reshape .and. alloc_in_loop) then
        write(*,*) "Error: alloc_in_loop cannot be .true. if explicit_reshape is .false."
        stop
      end if
    end if

    if (present(alloc_in_loop)) then
      write(*,'("Optional settings: alloc_in_loop=", L)') alloc_in_loop
    end if
    if (present(explicit_reshape)) then
      write(*,'("Optional settings: explicit_reshape=", L)') explicit_reshape
    end if

  end subroutine setup

  subroutine error_mesg(filename, line, message)

    use, intrinsic :: iso_fortran_env, only : stderr=>error_unit

    implicit none

    character(len=*), intent(in) :: filename, message
    integer,          intent(in) :: line

    write(stderr, '(a,":",I0, " - ", a)') filename, line, trim(adjustl(message))
    stop 100

  end subroutine error_mesg

end module utils
