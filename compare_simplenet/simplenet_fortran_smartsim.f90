program simplenet_fortran

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   use smartredis_client, only : client_type

   implicit none

   ! Set working precision for reals
   integer, parameter :: wp = sp

   ! Set up Fortran data structures
   real(wp), dimension(5) :: in_data
   real(wp), dimension(5) :: out_data

   type(client_type) :: client
   integer :: ierr

   ! Initialise data
   in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
   write(*,*) "Input (Fortran):", in_data

   ierr = client%initialize(.false.)
   ! if (ierr /= SRNoError) error stop  ! TODO: import SRNoError
   ierr = client%put_tensor("in_data", in_data, shape(in_data))
   ! if (ierr /= SRNoError) error stop  ! TODO: import SRNoError

   ! call sleep(10)

   ierr = client%unpack_tensor("out_data", out_data, shape(out_data))
   ! if (ierr /= SRNoError) error stop  ! TODO: import SRNoError

   write(*,*) "Output (Fortran):", out_data

end program simplenet_fortran
