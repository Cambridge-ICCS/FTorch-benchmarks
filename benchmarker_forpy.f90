program benchmarker
use forpy_mod

integer ie

ie = forpy_initialize()
if (ie .ne. 0) then
    write(*,*)'Forpy did not initialize, stopping'
    stop
endif

! Read in saved input (and output) values

! Initialise the PyTorch model

! Start timing

! Run inference X many times (check output)

! Stop timing, output results
end program benchmarker
