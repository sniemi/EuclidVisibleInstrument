!MODULE CDM03

SUBROUTINE CDM03(sinp,xdim,ydim,sout,iflip,jflip,dob,rdose,zdim,in_nt,in_sigma,in_tr)
!------------------------------------------------------------------------------
! 
! Radiation damage model CDM03.
! 
!------------------------------------------------------------------------------

USE nrtype

implicit none

integer, parameter :: kdim=7
integer, intent(in) :: xdim,ydim,zdim,iflip,jflip
double precision, intent(in) :: in_nt(zdim),in_sigma(zdim),in_tr(zdim)
real, dimension(xdim,ydim), intent(in)  :: sinp
real, dimension(xdim,ydim), intent(out) :: sout
double precision, intent(in) :: dob,rdose
integer, allocatable :: ci(:)
double precision, allocatable :: s(:,:),no(:,:),sno(:,:)
double precision, allocatable :: slsf(:),stot(:),lsf(:)
integer :: i,j,k
double precision :: x=0.
double precision :: y=0.
double precision :: beta=0.6
double precision :: fwc=175000.
double precision :: vth=1.168e7
double precision :: t=1.024e-2
double precision :: vg=6.e-11
double precision :: nc,nr
double precision :: st=5.e-6
double precision :: sfwc=730000.
double precision :: svg=2.5e-10
double precision, allocatable :: alpha(:),gamm(:),g(:)

!relative density and release times for -110 degrees [from Gordon Hopkinson final report for Euclid]
!real(sp), dimension(kdim) :: nt=(/5.0,0.22,0.2,0.1,0.043,0.39,1.0/)
!real(sp), dimension(kdim) :: tr=(/0.0000003,0.0001,0.002,0.025,0.124,4.5,64.1/)

!relative density and release times for -120 degrees [from Gordon Hopkinson final report for Euclid]
double precision, dimension(kdim) :: nt=(/5.0,0.22,0.2,0.1,0.043,0.39,1.0/)
double precision, dimension(kdim) :: tr=(/0.00000082,0.0003,0.002,0.025,0.124,16.7,496.0/)

!capture cross sections from Gaia data
!assumed to correspond with traps seen in Euclid testing
double precision, dimension(kdim) :: sigma=(/2.2e-13,2.2e-13,4.72e-15,1.37e-16,2.78e-17,1.93e-17,6.39e-18/)

!allocate memory
allocate(ci(xdim),lsf(xdim),slsf(ydim),stot(ydim))
allocate(s(xdim,ydim))
allocate(no(ydim,kdim),sno(xdim,kdim))
allocate(alpha(kdim),gamm(kdim),g(kdim))

!set up variables
ci=0
s=0.
no=0.
sno=0.
stot=0.
st=5.e-6
t=st*2048.

!absolute trap density which should be scaled according to radiation dose
!(nt=1.5e10 gives approx fit to GH data for a dose of 8e9 10MeV equiv. protons)
nt=nt*rdose
nt=in_nt*rdose
sigma=in_sigma
tr=in_tr

do i=1,xdim
   do j=1,ydim
      s(i,j)=sinp(i+iflip*(xdim+1-2*i),j+jflip*(ydim+1-2*j))
   enddo
enddo

!add some background electrons
s=s+dob

!apply FWC (anti-blooming)
s=min(s,fwc)

!parallel direction
alpha=t*sigma*vth*fwc**beta/2./vg
g=nt*2.*vg/fwc**beta

do i=1,xdim
   gamm=g*(x+real(i))
   do k=1,kdim
      do j=1,ydim
         nc=0.
         
         if(s(i,j).gt.0.0)then
           nc=max((gamm(k)*s(i,j)**beta-no(j,k))/(gamm(k)*s(i,j)**(beta-1.)+1.)*(1.-exp(-alpha(k)*s(i,j)**(1.-beta))),0.d0)
         endif

         s(i,j)=s(i,j)-nc
         no(j,k)=no(j,k)+nc
         nr=no(j,k)*(1.-exp(-t/tr(k)))
         s(i,j)=s(i,j)+nr
         no(j,k)=no(j,k)-nr
      enddo
   enddo
enddo

!serial direction
alpha=st*sigma*vth*sfwc**beta/2./svg
g=nt*2.*svg/sfwc**beta

do j=1,ydim
   gamm=g*(y+real(j))
   do k=1,kdim
      if(tr(k).lt.t)then
         do i=1,xdim
            nc=0.
            
            if(s(i,j).gt.0.01)then
              nc=max((gamm(k)*s(i,j)**beta-sno(i,k))/(gamm(k)*s(i,j)**(beta-1.)+1.)*(1.-exp(-alpha(k)*s(i,j)**(1.-beta))),0.d0)
            endif

            s(i,j)=s(i,j)-nc
            sno(i,k)=sno(i,k)+nc
            nr=sno(i,k)*(1.-exp(-st/tr(k)))
            s(i,j)=s(i,j)+nr
            sno(i,k)=sno(i,k)-nr
         enddo
      endif
   enddo
enddo

do i=1,xdim
   do j=1,ydim
      sout(i+iflip*(xdim+1-2*i),j+jflip*(ydim+1-2*j)) = s(i,j)
   enddo
enddo

deallocate(ci,s,no,sno,slsf,stot,lsf)
deallocate(alpha,gamm,g)

END SUBROUTINE cdm03
!END MODULE CDM03
