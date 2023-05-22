program benchmarker
use forpy_mod

integer ie

ie = forpy_initialize()
if (ie .ne. 0) then
    write(*,*)'Forpy did not initialize, stopping'
    stop
endif
end program benchmarker
