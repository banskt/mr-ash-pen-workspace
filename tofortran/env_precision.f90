MODULE env_precision
    IMPLICIT NONE
    INTEGER, PARAMETER :: r4_kind  = SELECTED_REAL_KIND(6, 37)
    INTEGER, PARAMETER :: r8_kind  = SELECTED_REAL_KIND(15, 307)
    INTEGER, PARAMETER :: r16_kind = SELECTED_REAL_KIND(33, 4931)
    INTEGER, PARAMETER :: i2_kind  = SELECTED_INT_KIND(4)
    INTEGER, PARAMETER :: i4_kind  = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER :: i8_kind  = SELECTED_INT_KIND(15)

END MODULE env_precision
