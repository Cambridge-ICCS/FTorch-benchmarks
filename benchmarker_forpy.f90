program benchmarker
use forpy_mod
use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
                                          stdout=>output_unit, &
                                          stderr=>error_unit
use cg_drag_forpy_mod, only: cg_drag_ML_init, cg_drag_ML_end, cg_drag_ML
implicit none

integer :: ie, ntimes, i, j, k, ii, jj, kk, iter
character(len=10) :: ntimes_char
character(len=1024) :: model_dir, model_name
real(kind=8), dimension(:,:,:), allocatable :: uuu, vvv, gwfcng_x, gwfcng_y
real :: start_time, end_time
real(kind=8), dimension(:,:), allocatable :: lat, psfc
integer, parameter :: I_MAX=128, J_MAX=64, K_MAX=40

ie = forpy_initialize()

! Parse argument for N
if (command_argument_count() .ne. 3) then
    write(stderr, *)'Usage: benchmarker <model-dir> <model-name> <N>'
    write(stderr, *)'       Run model named <model-name> in <model-dir> N'
    write(stderr, *)'       times with forpy'
    stop
endif
call get_command_argument(1, model_dir)
call get_command_argument(2, model_name)
call get_command_argument(3, ntimes_char)
read(ntimes_char, *) ntimes
write(*,*) 'Will run ', trim(model_dir),' ', ntimes, ' times'

allocate(uuu(I_MAX, J_MAX, K_MAX))
allocate(vvv(I_MAX, J_MAX, K_MAX))
allocate(gwfcng_x(I_MAX, J_MAX, K_MAX))
allocate(gwfcng_y(I_MAX, J_MAX, K_MAX))
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

! Initialise the model
call cg_drag_ML_init(model_dir, model_name)

! Start timing
call cpu_time(start_time)

! Run inference N many times (check output)
do iter = 1, ntimes
    call cg_drag_ML(uuu, vvv, psfc, lat, gwfcng_x, gwfcng_y)
end do

! Stop timing, output results
call cpu_time(end_time)
write(*,*)'Time taken: ', end_time-start_time

! Clean up
call cg_drag_ML_end
deallocate(uuu)
deallocate(vvv)
deallocate(gwfcng_x)
deallocate(gwfcng_y)
deallocate(lat)
deallocate(psfc)
end program benchmarker
