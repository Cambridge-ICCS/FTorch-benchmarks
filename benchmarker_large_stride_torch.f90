program benchmark_stride_test

use, intrinsic :: iso_c_binding
use ftorch

implicit none

! Make a large array of 100,000,000 elements.
integer, parameter :: ni = 10000, nj = 10000 
integer :: i
integer(c_int), parameter :: n_inputs = 1
integer(c_int64_t) :: shape_2d(2)
integer(c_int) :: stride_2d(2)
real, dimension(:,:), allocatable, target :: big_array, big_result

type(torch_tensor) :: result_tensor
type(torch_tensor), dimension(n_inputs), target :: input_array
type(torch_module) :: model

model = torch_module_load("saved_model.pth")
allocate(big_array(ni, nj))
allocate(big_result(ni, nj))

! Fill the array.
call random_number(big_array)

shape_2d = (/ ni, nj /)
stride_2d = (/ 1, 2 /)
! Create input and output tensors for the model.
input_array(1) = torch_tensor_from_blob(c_loc(big_array), 2, shape_2d, torch_kFloat32, torch_kCPU, stride_2d)
result_tensor = torch_tensor_from_blob(c_loc(big_result), 2, shape_2d, torch_kFloat32, torch_kCPU, stride_2d)

call torch_module_forward(model, input_array, n_inputs, result_tensor)

! Clean up.
call torch_tensor_delete(result_tensor)
do i = 1, n_inputs
  call torch_tensor_delete(input_array(i))
end do
call torch_module_delete(model)

end program

