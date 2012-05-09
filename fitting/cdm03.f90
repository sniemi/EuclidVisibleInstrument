!MODULE CDM03

SUBROUTINE CDM03(sinp,xdim,ydim,sout,iflip,jflip,dob,rdose,zdim,in_nt,in_sigma,in_tr)
!------------------------------------------------------------------------------
! 
! Radiation damage model CDM03, see EUCLID_TN_ESA_AS_003_0-2.pdf for details.
! 
!------------------------------------------------------------------------------
! dob = Diffuse (long term) Optical Background [e-/pixel/transit]
! zdim = number of trap species

USE nrtype

IMPLICIT NONE

INTEGER, INTENT(in) 					:: xdim,ydim,zdim,iflip,jflip
DOUBLE PRECISION, INTENT(in) 			:: in_nt(zdim),in_sigma(zdim),in_tr(zdim)
REAL, DIMENSION(xdim,ydim), INTENT(in)  :: sinp
REAL, DIMENSION(xdim,ydim), INTENT(out) :: sout
DOUBLE PRECISION, INTENT(in) 			:: dob,rdose
INTEGER, ALLOCATABLE 					:: ci(:)
DOUBLE PRECISION, ALLOCATABLE 			:: s(:,:),no(:,:),sno(:,:)
DOUBLE PRECISION, ALLOCATABLE 			:: slsf(:),stot(:),lsf(:)
INTEGER 								:: i,j,k
DOUBLE PRECISION 						:: x, y

!model related variables
DOUBLE PRECISION :: nc,nr				! number of electrons captured, released
DOUBLE PRECISION :: beta=0.6			! charge cloud expansion parameter [0, 1]
DOUBLE PRECISION :: fwc=175000.			! full well capacity
DOUBLE PRECISION :: vth=1.168e7			! electron thermal velocity [cm/s]
DOUBLE PRECISION :: t=1.024e-2			! parallel line time [s]
DOUBLE PRECISION :: vg=6.e-11			! geometric confinement volume
DOUBLE PRECISION :: st=5.e-6			! serial pixel transfer period [s]
DOUBLE PRECISION :: sfwc=730000.		! serial (readout register) pixel full well capacity
DOUBLE PRECISION :: svg=2.5e-10			! geometrical confinement volume of serial register pixels [cm**3]

DOUBLE PRECISION, ALLOCATABLE :: alpha(:),gamm(:),g(:)

!If uncommented, parameters are set to defaults
!relative density and release times for -120 degrees [from Gordon Hopkinson final report for Euclid]
!DOUBLE PRECISION, DIMENSION(zdim) :: nt =(/5.0,0.22,0.2,0.1,0.043,0.39,1.0/)
DOUBLE PRECISION, DIMENSION(zdim) :: nt 

DOUBLE PRECISION, DIMENSION(zdim) :: tr ! =(/0.00000082,0.0003,0.002,0.025,0.124,16.7,496.0/)
!capture cross sections from Gaia data, assumed to correspond with traps seen in Euclid testing
DOUBLE PRECISION, DIMENSION(zdim) :: sigma ! =(/2.2e-13,2.2e-13,4.72e-15,1.37e-16,2.78e-17,1.93e-17,6.39e-18/)

!allocate memory
ALLOCATE(ci(xdim),lsf(xdim),slsf(ydim),stot(ydim))
ALLOCATE(s(xdim,ydim))
ALLOCATE(no(ydim,zdim),sno(xdim,zdim))
ALLOCATE(alpha(zdim),gamm(zdim),g(zdim))

!set up variables
x = 0.
y = 0.
ci = 0
s = 0.
no = 0.
sno = 0.
stot = 0.

!absolute trap density which should be scaled according to radiation dose
!(nt=1.5e10 gives approx fit to GH data for a dose of 8e9 10MeV equiv. protons)
nt=in_nt*rdose			!absolute trap density [per cm**3]
sigma=in_sigma
tr=in_tr

! flip data for Euclid depending on the quadrant being processed
DO i=1,xdim
   DO j=1,ydim
      s(i,j)=sinp(i+iflip*(xdim+1-2*i),j+jflip*(ydim+1-2*j))
   ENDDO
ENDDO

!add some background electrons
s=s+dob

!apply FWC (anti-blooming)
s=min(s,fwc)

!parallel direction
alpha=t*sigma*vth*fwc**beta/2./vg
g=nt*2.*vg/fwc**beta

DO i=1,xdim
   gamm = g * (x + REAL(i))
   DO k=1,zdim
      DO j=1,ydim
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

!serial direction
alpha=st*sigma*vth*sfwc**beta/2./svg
g=nt*2.*svg/sfwc**beta

DO j=1,ydim
   gamm = g * ( y + REAL(j))
   DO k=1,zdim
      IF(tr(k).lt.t)THEN
         DO i=1,xdim
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

! flip data back to the input orientation
DO i=1,xdim
   DO j=1,ydim
      sout(i+iflip*(xdim+1-2*i),j+jflip*(ydim+1-2*j)) = s(i,j)
   ENDDO
ENDDO

! free memory
DEALLOCATE(ci,s,no,sno,slsf,stot,lsf)
DEALLOCATE(alpha,gamm,g)

END SUBROUTINE cdm03
!END MODULE CDM03
