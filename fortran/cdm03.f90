SUBROUTINE CDM03(sinp,xdim,ydim,sout,iflip,jflip,dob,rdose,zdim,in_nt,in_sigma,in_tr)
!------------------------------------------------------------------------------
! 
! Radiation damage model CDM03, see EUCLID_TN_ESA_AS_003_0-2.pdf for details.
! 
! Modified to work with Euclid VIS instrument by Sami-Matias Niemi
!	Note that the coordinate system in Gaia (original CDM03 model) and
!   VIS differ. In VIS the serial register is at the "bottom" while in 
!   Gaia it is on the "left" side of the CCD quad. 
!
! Note: This version is intended to be called from f2py which does not do
!       well with allocatable arrays.
!
!------------------------------------------------------------------------------
! ARGUMENTS:
! sinp = input image array
! xdim = dimension of the input array in x direction (parallel)
! ydim = dimension of the input array in y direction (serial)
! sout = output image array with added CTI
! dob = Diffuse (long term) Optical Background [e-/pixel/transit]
! zdim = number of trap species
!------------------------------------------------------------------------------

USE nrtype

IMPLICIT NONE

!inputs, outputs and work variables
INTEGER, INTENT(in) :: xdim, ydim, zdim, iflip, jflip
REAL, INTENT(in) :: in_nt(zdim),in_sigma(zdim),in_tr(zdim)
REAL, DIMENSION(xdim,ydim), INTENT(in)  :: sinp
REAL, DIMENSION(xdim,ydim), INTENT(out) :: sout
REAL, INTENT(in) :: dob,rdose
REAL :: no(ydim,zdim),sno(ydim,zdim)
REAL :: s(ydim, ydim)
INTEGER :: i,j,k

!model related variables
DOUBLE PRECISION :: nc,nr			! number of electrons captured, released
DOUBLE PRECISION :: beta=0.6			! charge cloud expansion parameter [0, 1]
DOUBLE PRECISION :: fwc=175000.			! full well capacity
DOUBLE PRECISION :: vth=1.168e7			! electron thermal velocity [cm/s]
DOUBLE PRECISION :: t=1.024e-2			! parallel line time [s] for 200kHz
!DOUBLE PRECISION :: t=2.07e-3			! parallel line time [s] for 1MHz
DOUBLE PRECISION :: vg=6.e-11			! geometric confinement volume
DOUBLE PRECISION :: st=5.e-6			! serial pixel transfer period [s] for 200kHz
!DOUBLE PRECISION :: st=2.07e-3 / 2048.         ! serial pixel transfer period [s] for 1MHz
DOUBLE PRECISION :: sfwc=730000.		! serial (readout register) pixel full well capacity
DOUBLE PRECISION :: svg=1.0e-10			! geometrical confinement volume of serial register pixels [cm**3]

!Trap parameters
DOUBLE PRECISION, DIMENSION(7) :: nt, tr, sigma
DOUBLE PRECISION, DIMENSION(zdim):: alpha,gamm,g

! set up variables to zero
s = 0.
no = 0.
sno = 0.
sout = 0.

! absolute trap density which should be scaled according to radiation dose
! (nt=1.5e10 gives approx fit to GH data for a dose of 8e9 10MeV equiv. protons)
nt = in_nt * rdose				!absolute trap density [per cm**3]
sigma = in_sigma
tr = in_tr

! flip data for Euclid depending on the quadrant being processed and
! rotate (j, i slip in s) to move from Euclid to Gaia coordinate system
! because this is what is assumed in CDM03 (EUCLID_TN_ESA_AS_003_0-2.pdf)
DO i=1,xdim
   DO j=1,ydim
      s(j,i) = sinp(i+iflip*(xdim+1-2*i),j+jflip*(ydim+1-2*j))
   ENDDO
ENDDO

!add background electrons
s = s + dob

!apply FWC (anti-blooming)
s=min(s,fwc)

!Because of the transpose, we need to be careful what we now call
!xdim and ydim. In the following loops these have been changed
!not to exceed the array dimensions.

!start with parallel direction	 
alpha=t*sigma*vth*fwc**beta/2./vg
g=nt*2.*vg/fwc**beta

DO i=1,ydim
   gamm = g * REAL(i)
   DO k=1,zdim
      DO j=1,xdim
         nc=0.
         
         IF(s(i,j).gt.0.01)THEN
           nc=max((gamm(k)*s(i,j)**beta-no(j,k))/(gamm(k)*s(i,j)**(beta-1.)+1.)*(1.-exp(-alpha(k)*s(i,j)**(1.-beta))),0.d0)
         ENDIF

         no(j,k) = no(j,k) + nc
         nr = no(j,k) * (1. - exp(-t/tr(k)))
         s(i,j) = s(i,j) - nc + nr
         no(j,k) = no(j,k) - nr
      ENDDO
   ENDDO
ENDDO

!now serial direction
alpha=st*sigma*vth*sfwc**beta/2./svg
g=nt*2.*svg/sfwc**beta

DO j=1,xdim
   gamm = g * REAL(j)
   DO k=1,zdim
      IF(tr(k).lt.t)THEN
         DO i=1,ydim
            nc=0.
            
            IF(s(i,j).gt.0.01)THEN
              nc=max((gamm(k)*s(i,j)**beta-sno(i,k))/(gamm(k)*s(i,j)**(beta-1.)+1.)*(1.-exp(-alpha(k)*s(i,j)**(1.-beta))),0.d0)
            ENDIF

            sno(i,k) = sno(i,k) + nc
            nr = sno(i,k) * (1. - exp(-st/tr(k)))
            s(i,j) = s(i,j) - nc + nr
            sno(i,k) = sno(i,k) - nr
         ENDDO
      ENDIF
   ENDDO
ENDDO

! We need to rotate back from Gaia coordinate system and
! flip data back to the input orientation
DO i=1,xdim
   DO j=1,ydim
      sout(i+iflip*(xdim+1-2*i),j+jflip*(ydim+1-2*j)) = s(j,i)
   ENDDO
ENDDO

END SUBROUTINE cdm03
