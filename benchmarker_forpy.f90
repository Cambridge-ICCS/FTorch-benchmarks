program benchmarker
use forpy_mod
use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
                                          stdout=>output_unit, &
                                          stderr=>error_unit

integer :: ie, ntimes, i, j, k, ii, jj, kk
character(len=10) :: ntimes_char
real(kind=8), dimension(:,:,:), allocatable :: uuu, vvv
real(kind=8) :: val
real(kind=8), dimension(:,:), allocatable :: lat, psfc
integer, parameter :: I_MAX=128, J_MAX=64, K_MAX=40

ie = forpy_initialize()

! Parse argument for N
if (command_argument_count() .ne. 1) then
    write(stderr, *)'Must provide argument N, number of times to run inference'
    stop
endif
call get_command_argument(1, ntimes_char)
read(ntimes_char, *) ntimes
write(*,*) 'Will run inference ', ntimes, ' times'

allocate(uuu(I_MAX, J_MAX, K_MAX))
allocate(vvv(I_MAX, J_MAX, K_MAX))
allocate(lat(I_MAX, J_MAX))
allocate(psfc(I_MAX, J_MAX))
! Read in saved input (and output) values
open(10, file='../input_data/uuu.txt')
open(11, file='../input_data/vvv.txt')
open(12, file='../input_data/lat.txt')
open(13, file='../input_data/psfc.txt')
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

! Initialise the PyTorch model

! Start timing

! Run inference N many times (check output)

! Stop timing, output results
end program benchmarker
