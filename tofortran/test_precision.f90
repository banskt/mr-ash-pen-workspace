program test_precision
    use env_precision
    implicit none

    real(r4_kind) :: a
    real(r8_kind) :: b
    real(r16_kind) :: c
    integer(i2_kind) :: i
    integer(i4_kind) :: j
    integer(i8_kind) :: k

    character(len=*), parameter :: fmt1 = "(A7, F37.34, 2X, A6, I2, 2X, A11, I2)"
    character(len=*), parameter :: fmt2 = "(A7, I20, 2X, A6, I2, 2X, A11, I2)"

    write (*, *) "Enter input for a"
    read (*, *) a
    write (*, fmt1) "Input: ", a, "Kind: ", kind(a), "Precision: ", precision(a)

    write (*, *) "Enter input for b"
    read (*, *) b
    write (*, fmt1) "Input: ", b, "Kind: ", kind(b), "Precision: ", precision(b)

    write (*, *) "Enter input for c"
    read (*, *) c
    write (*, fmt1) "Input: ", c, "Kind: ", kind(c), "Precision: ", precision(c)

    write (*, *) "Enter input for i"
    read (*, *) i
    write (*, fmt2) "Input: ", i, "Kind: ", kind(i), "Bit size: ", bit_size(i)

    write (*, *) "Enter input for j"
    read (*, *) j
    write (*, fmt2) "Input: ", j, "Kind: ", kind(j), "Bit size: ", bit_size(j)

    write (*, *) "Enter input for k"
    read (*, *) k
    write (*, fmt2) "Input: ", k, "Kind: ", kind(k), "Bit size: ", bit_size(k)


end program test_precision
