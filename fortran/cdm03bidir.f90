SUBROUTINE CDM03(sinp,xdim,ydim,sout,iflip,jflip,dob,rdose,zdim_p,zdim_s,in_nt_p,in_sigma_p,in_tr_p,in_nt_s,in_sigma_s,in_tr_s)
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
INTEGER, INTENT(in)                                  :: xdim, ydim, zdim_p, zdim_s, iflip, jflip
DOUBLE PRECISION, DIMENSION(zdim_p), INTENT(in)      :: in_nt_p, in_sigma_p, in_tr_p
DOUBLE PRECISION, DIMENSION(zdim_s), INTENT(in)      :: in_nt_s, in_sigma_s, in_tr_s
DOUBLE PRECISION, DIMENSION(xdim, ydim), INTENT(in)  :: sinp
DOUBLE PRECISION, DIMENSION(xdim, ydim), INTENT(out) :: sout
DOUBLE PRECISION, INTENT(in)                         :: dob, rdose

!work variables
DOUBLE PRECISION, ALLOCATABLE                        :: no(:,:), sno(:,:), s(:,:)
DOUBLE PRECISION                                     :: nc,nr               ! number of electrons captured, released
INTEGER                                              :: i, j, k

!CDM03 model related variables
!MSSL parameters when fitting to test data
DOUBLE PRECISION :: beta_p=0.29         ! charge cloud expansion parameter in parallel direction [0, 1]
DOUBLE PRECISION :: beta_s=0.12         ! charge cloud expansion parameter in serial direction [0, 1]
DOUBLE PRECISION :: fwc=200000.         ! full well capacity
DOUBLE PRECISION :: vth=1.168e7         ! electron thermal velocity [cm/s]
DOUBLE PRECISION :: t=20.48e-3          ! parallel line time [s] (Hopkinson test data)
DOUBLE PRECISION :: vg=6.e-11           ! geometric confinement volume (MSSL)
DOUBLE PRECISION :: st=5.0e-6           ! serial pixel transfer period [s] for (Hopkinson test data)
DOUBLE PRECISION :: sfwc=730000.        ! serial (readout register) pixel full well capacity
DOUBLE PRECISION :: svg=1.0e-10         ! geometrical confinement volume of serial register pixels [cm**3] (MSSL)

!MSSL parameter values when simulating
!DOUBLE PRECISION :: beta_p=0.6          ! charge cloud expansion parameter in parallel direction [0, 1]
!DOUBLE PRECISION :: beta_s=0.6          ! charge cloud expansion parameter in serial direction [0, 1]
!DOUBLE PRECISION :: fwc=200000.         ! full well capacity
!DOUBLE PRECISION :: vth=1.168e7         ! electron thermal velocity [cm/s]
!DOUBLE PRECISION :: t=1.0e-3            ! parallel line time [s] (NEW))
!DOUBLE PRECISION :: vg=7.20E-11         ! geometric confinement volume
!DOUBLE PRECISION :: st=1.428e-5         ! serial pixel transfer period [s] for 70kHz
!DOUBLE PRECISION :: sfwc=200000.        ! serial (readout register) pixel full well capacity
!DOUBLE PRECISION :: svg=1.20E-10        ! geometrical confinement volume of serial register pixels [cm**3] (MSSL)


!Thibaut's parameter values
!DOUBLE PRECISION :: beta_p=0.29         ! charge cloud expansion parameter in parallel direction [0, 1]
!DOUBLE PRECISION :: beta_s=0.12         ! charge cloud expansion parameter in serial direction [0, 1]
!DOUBLE PRECISION :: fwc=200000.         ! full well capacity
!DOUBLE PRECISION :: vth=1.62E+07        ! electron thermal velocity [cm/s]
!DOUBLE PRECISION :: t=2.10E-02          ! parallel line time [s] (Hopkinson test data)
!DOUBLE PRECISION :: vg=7.20E-11         ! geometric confinement volume (Thibaut)
!DOUBLE PRECISION :: st=5.00E-06         ! serial pixel transfer period [s] for (Hopkinson test data)
!DOUBLE PRECISION :: sfwc=1450000.       ! serial (readout register) pixel full well capacity
!DOUBLE PRECISION :: svg=3.00E-10        ! geometrical confinement volume of serial register pixels [cm**3] (Thibaut)

! helper
INTEGER :: zdim

! trap related variables, parallel (_p) and serial direction (_s)
DOUBLE PRECISION, DIMENSION(zdim_p)   :: nt_p, tr_p, sigma_p, gamm_p, g_p, alpha_p
DOUBLE PRECISION, DIMENSION(zdim_s)   :: nt_s, tr_s, sigma_s, gamm_s, g_s, alpha_s

zdim = max(zdim_p, zdim_s)

!reserve space based on the longer dimension
IF (xdim > ydim) THEN
  ALLOCATE(s(xdim, xdim), sno(xdim, zdim), no(xdim, zdim))
ELSE
  ALLOCATE(s(ydim, ydim), sno(ydim, zdim), no(ydim, zdim))
ENDIF


! set up variables to zero
s(:,:) = 0.
no(:,:) = 0.
sno(:,:) = 0.
sout(:,:) = 0.

! absolute trap density which should be scaled according to radiation dose
! (nt=1.5e10 gives approx fit to GH data for a dose of 8e9 10MeV equiv. protons)
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
      s(j, i) = sinp(i+iflip*(xdim+1-2*i), j+jflip*(ydim+1-2*j))
   ENDDO
ENDDO

!add background electrons
s = s + dob

!apply FWC (anti-blooming)
s = min(s, fwc)

!Because of the transpose, we need to be careful what we now call
!xdim and ydim. In the following loops these have been changed
!not to exceed the array dimensions.

!start with parallel direction	 
alpha_p=t*sigma_p*vth*fwc**beta_p/2./vg
g_p=nt_p*2.*vg/fwc**beta_p

DO i = 1, ydim
   gamm_p = g_p * REAL(i)
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

!now serial direction
alpha_s=st*sigma_s*vth*sfwc**beta_s/2./svg
g_s=nt_s*2.*svg/sfwc**beta_s

DO j = 1, ydim
   gamm_s = g_s * REAL(j)
   DO k=1, zdim_s
      IF(tr_s(k).lt.t)THEN
         DO i = 1, ydim
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
      ENDIF
   ENDDO
ENDDO

! We need to rotate back from Gaia coordinate system and
! flip data back to the input orientation
DO i = 1, xdim
   DO j = 1, ydim
      sout(i+iflip*(xdim+1-2*i), j+jflip*(ydim+1-2*j)) = s(j, i)
   ENDDO
ENDDO

DEALLOCATE(s, sno, no)

END SUBROUTINE cdm03
