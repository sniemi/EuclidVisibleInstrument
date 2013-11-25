SUBROUTINE CDM03(sinp,xdim,ydim,sout,iflip,jflip,dob,rdose,zdim_p,zdim_s,params, &
                 in_nt_p,in_sigma_p,in_tr_p,in_nt_s,in_sigma_s,in_tr_s,in_params)
!------------------------------------------------------------------------------------
! 
! Radiation damage model CDM03, see EUCLID_TN_ESA_AS_003_0-2.pdf for details.
!
! This version allows different trap parameters to be used in parallel and serial direction.
!
! Modified to work with Euclid VIS instrument by Sami-Matias Niemi.
!
! Note that the coordinate system in Gaia (original CDM03 model) and
! VIS differ. In VIS the serial register is at the "bottom" while in
! Gaia it is on the "left" side of the CCD quad.
!
! Note: This version is intended to be called from f2py.
!
!------------------------------------------------------------------------------------
! ARGUMENTS:
! sinp = input image array
! xdim = dimension of the input array in x direction (parallel)
! ydim = dimension of the input array in y direction (serial)
! sout = output image array with added CTI
! dob = Diffuse (long term) Optical Background [e-/pixel/transit]
! zdim = number of trap species
!------------------------------------------------------------------------------------
IMPLICIT NONE

!inputs and outputs
INTEGER, INTENT(in)                                  :: xdim, ydim, zdim_p, zdim_s, iflip, jflip, params
DOUBLE PRECISION, DIMENSION(zdim_p), INTENT(in)      :: in_nt_p, in_sigma_p, in_tr_p
DOUBLE PRECISION, DIMENSION(zdim_s), INTENT(in)      :: in_nt_s, in_sigma_s, in_tr_s
DOUBLE PRECISION, DIMENSION(params), INTENT(in)      :: in_params
DOUBLE PRECISION, DIMENSION(xdim, ydim), INTENT(in)  :: sinp
DOUBLE PRECISION, DIMENSION(xdim, ydim), INTENT(out) :: sout
DOUBLE PRECISION, INTENT(in)                         :: dob, rdose

!work variables
DOUBLE PRECISION, ALLOCATABLE                        :: no(:,:), sno(:,:), s(:,:)
DOUBLE PRECISION                                     :: nc,nr               ! number of electrons captured, released
INTEGER                                              :: i, j, k

!CDM03 model related variables
DOUBLE PRECISION :: beta_p     ! charge cloud expansion parameter in parallel direction [0, 1]
DOUBLE PRECISION :: beta_s     ! charge cloud expansion parameter in serial direction [0, 1]
DOUBLE PRECISION :: fwc        ! full well capacity
DOUBLE PRECISION :: vth        ! electron thermal velocity [cm/s]
DOUBLE PRECISION :: t          ! parallel line time [s] (Hopkinson test data)
DOUBLE PRECISION :: vg         ! geometric confinement volume
DOUBLE PRECISION :: st         ! serial pixel transfer period [s] for (Hopkinson test data)
DOUBLE PRECISION :: sfwc       ! serial (readout register) pixel full well capacity
DOUBLE PRECISION :: svg        ! geometrical confinement volume of serial register pixels [cm**3]

! helper
INTEGER :: zdim
DOUBLE PRECISION :: parallel
DOUBLE PRECISION :: serial

! trap related variables, parallel (_p) and serial direction (_s)
DOUBLE PRECISION, DIMENSION(zdim_p)   :: nt_p, tr_p, sigma_p, gamm_p, g_p, alpha_p
DOUBLE PRECISION, DIMENSION(zdim_s)   :: nt_s, tr_s, sigma_s, gamm_s, g_s, alpha_s

zdim = max(zdim_p, zdim_s)

!reserve space based on the longer dimension
ALLOCATE(s(xdim, ydim), sno(xdim, zdim), no(ydim, zdim))

beta_p = in_params(1)
beta_s = in_params(2)
fwc = in_params(3)
vth = in_params(4)
vg = in_params(5)
t = in_params(6)
sfwc = in_params(7)
svg = in_params(8)
st = in_params(9)
parallel = in_params(10)
serial = in_params(11)

!PRINT *, beta_p, beta_s, fwc, vth, vg, t, sfwc, svg, st, parallel, serial

! set up variables to zero
s(:,:) = 0.
no(:,:) = 0.
sno(:,:) = 0.
sout(:,:) = 0.

! trap density should be scaled according to radiation dose
nt_p = in_nt_p * rdose                    !absolute trap density [per cm**3]
sigma_p = in_sigma_p
tr_p = in_tr_p
nt_s = in_nt_s * rdose                    !absolute trap density [per cm**3]
sigma_s = in_sigma_s
tr_s = in_tr_s

! flip data for Euclid depending on the quadrant being processed and
! rotate (j, i slip in s) to move from Euclid to Gaia coordinate system
! because this is what is assumed in CDM03 (EUCLID_TN_ESA_AS_003_0-2.pdf)
DO i = 1, xdim
   DO j = 1, ydim
      s(i, j) = sinp(i+iflip*(xdim+1-2*i), j+jflip*(ydim+1-2*j))
   ENDDO
ENDDO

!add background electrons
s = s + dob

!apply FWC (anti-blooming)
s = min(s, fwc)

IF (parallel > 0.) THEN
    !parallel direction
    PRINT *, "Including parallel CTI"
    alpha_p=t*sigma_p*vth*fwc**beta_p/2./vg
    g_p=nt_p*2.*vg/fwc**beta_p
    !g_p = 0.022360679774997897 * 2.   !fix this to get agreement with thibaut's test results, a single source
    PRINT *, g_p
    PRINT *, alpha_p

    DO i = 1, xdim
       gamm_p = g_p * (REAL(i) - 1)  !had to include -1 to get agreement with Thibaut, bug in Java version?
       DO k = 1, zdim_p
          DO j = 1, ydim
             nc=0.

             IF(s(i,j).gt.0.01)THEN
               nc=max((gamm_p(k)*s(i,j)**beta_p-no(j,k))/(gamm_p(k)*s(i,j)**(beta_p-1.)+1.) &
               *(1.-exp(-alpha_p(k)*s(i,j)**(1.-beta_p))),0.d0)
             ENDIF

             no(j,k) = no(j,k) + nc
             nr = no(j,k) * (1. - exp(-t/tr_p(k)))
             s(i,j) = s(i,j) - nc + nr
             no(j,k) = no(j,k) - nr
          ENDDO
       ENDDO
    ENDDO
ENDIF

IF (serial > 0.) THEN
    !serial direction
    PRINT *, "Including serial CTI"
    alpha_s=st*sigma_s*vth*sfwc**beta_s/2./svg
    !g_s=nt_s*2.*svg/sfwc**beta_s !usual equation
    g_s=nt_s*2.*vg/sfwc**beta_s  !because of the way Thibaut gives the traps
    !g_s = 0.022360679774997897  * 2. !the value needed
    PRINT *, g_s
    print *, alpha_s

    DO j = 1, ydim
       gamm_s = g_s * REAL(j)
       DO k=1, zdim_s
         DO i = 1, xdim
            nc=0.

            IF(s(i,j).gt.0.01)THEN
              nc=max((gamm_s(k)*s(i,j)**beta_s-sno(i,k))/(gamm_s(k)*s(i,j)**(beta_s-1.)+1.) &
              *(1.-exp(-alpha_s(k)*s(i,j)**(1.-beta_s))),0.d0)
            ENDIF

            sno(i,k) = sno(i,k) + nc
            nr = sno(i,k) * (1. - exp(-st/tr_s(k)))
            s(i,j) = s(i,j) - nc + nr
            sno(i,k) = sno(i,k) - nr
         ENDDO
       ENDDO
    ENDDO
ENDIF

! flip data back to the input orientation
DO i = 1, xdim
   DO j = 1, ydim
      sout(i+iflip*(xdim+1-2*i), j+jflip*(ydim+1-2*j)) = s(i, j)
   ENDDO
ENDDO

DEALLOCATE(s, sno, no)

END SUBROUTINE cdm03
