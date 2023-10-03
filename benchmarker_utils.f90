module utils

  use, intrinsic :: iso_fortran_env, only : dp => real64

  implicit none

  interface print_time_stats
    module procedure print_time_stats_real, print_time_stats_dp
  end interface

contains

  subroutine assert_real_2d(a, b, test_name, rtol_opt)

    implicit none

    character(len=*) :: test_name
    real, intent(in), dimension(:,:) :: a, b
    real, optional :: rtol_opt
    real :: relative_error, rtol

    character(len=15) :: pass, fail

    fail = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
    pass = char(27)//'[32m'//'PASSED'//char(27)//'[0m'

    if (.not. present(rtol_opt)) then
      rtol = 1e-5
    else
      rtol = rtol_opt
    end if

    relative_error = maxval(abs(a/b - 1.))

    if (relative_error > rtol) then
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') fail, trim(test_name), relative_error
    else
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') pass, trim(test_name), relative_error
    end if

  end subroutine assert_real_2d

  subroutine assert_real(a, b, test_name, rtol_opt)

    implicit none

    character(len=*) :: test_name
    real, intent(in) :: a, b
    real, optional :: rtol_opt
    real :: relative_error, rtol

    character(len=15) :: pass, fail

    fail = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
    pass = char(27)//'[32m'//'PASSED'//char(27)//'[0m'

    if (.not. present(rtol_opt)) then
      rtol = 1e-5
    else
      rtol = rtol_opt
    end if

    relative_error = abs(a/b - 1.)

    if (relative_error > rtol) then
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') fail, trim(test_name), relative_error
    else
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') pass, trim(test_name), relative_error
    end if

  end subroutine assert_real

  subroutine print_time_stats_real(durations)

    implicit none

    real, intent(in) :: durations(:)
    integer :: i, n
    real :: mean, var, stddev

    ! skip the first element because this is always slower
    n = size(durations(2:), 1)

    mean = sum(durations(2:)) / n

    var = 0.

    do i = 2, n
      var = var + ( (durations(i) - mean)**2 / (n - 1) ) ! (n - 1) here is for corrected sample standard deviation
    end do

    stddev = sqrt(var)

    write(*,'(A,F10.4)') "min    time taken (s): ", minval(durations(2:))
    write(*,'(A,F10.4)') "max    time taken (s): ", maxval(durations(2:))
    write(*,'(A,F10.4)') "mean   time taken (s): ", mean
    write(*,'(A,F10.4)') "stddev time taken (s): ", stddev
    write(*,'(A,I10)')   "sample size          : ", n

  end subroutine print_time_stats_real

  subroutine print_time_stats_dp(durations)

    implicit none

    double precision, intent(in) :: durations(:)
    integer :: i, n
    double precision :: mean, var, stddev

    ! skip the first element because this is always slower
    n = size(durations(2:), 1)

    mean = sum(durations(2:)) / n

    var = 0._dp

    do i = 2, n
      var = var + ( (durations(i) - mean)**2._dp / (n - 1._dp) ) ! (n - 1) here is for corrected sample standard deviation
    end do

    stddev = sqrt(var)

    write(*,'(A,F10.4,A)') "min    time taken (s): ", minval(durations(2:)), " [omp]"
    write(*,'(A,F10.4,A)') "max    time taken (s): ", maxval(durations(2:)), " [omp]"
    write(*,'(A,F10.4,A)') "mean   time taken (s): ", mean, " [omp]"
    write(*,'(A,F10.4,A)') "stddev time taken (s): ", stddev, " [omp]"
    write(*,'(A,I10)')     "sample size          : ", n

  end subroutine print_time_stats_dp

  subroutine print_array_2d(array)

    implicit none

    real, intent(in) :: array(:,:)
    integer :: i

    do i = lbound(array,1), ubound(array,1)
      write (*, *) array(i,:)
    end do

  end subroutine print_array_2d

  subroutine setup(model_dir, model_name, ntimes, n)

    implicit none

    character(len=:), allocatable, intent(inout) :: model_dir, model_name
    integer, intent(out) :: ntimes, n

    character(len=1024) :: model_dir_temp, model_name_temp
    character(len=16) :: ntimes_char, n_char

    ! Parse argument for N
    if (command_argument_count() .ne. 4) then
      call error_mesg(__FILE__, __LINE__, "Usage: benchmarker <model-dir> <model-name> <ntimes> <N>")
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

  end subroutine setup

  subroutine error_mesg (filename, line, message)

    use, intrinsic :: iso_fortran_env, only : stderr=>error_unit

    implicit none

    character(len=*), intent(in) :: filename, message
    integer,          intent(in) :: line

    write(stderr, '(a,":",I0, " - ", a)') filename, line, trim(adjustl(message))
    stop 100

  end subroutine error_mesg

end module utils
