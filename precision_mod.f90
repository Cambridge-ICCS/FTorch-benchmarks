module precision

  use, intrinsic :: iso_fortran_env, only : sp => real32, dp => real64
  ! Imports primitives used to interface with C
  use, intrinsic :: iso_c_binding, only: c_sp=>c_float, c_dp=>c_double

  implicit none

  integer, parameter :: c_wp = c_sp
  integer, public, parameter :: wp = sp

end module precision
