function [T,Y]=ThreeLevelOBE(Detune1,Detune2,RabiFrq1,RabiFrq2,Tmax,PLOT_BOOL)
%function [T,Y]=ThreeLevelOBE(Detune1,Detune2,RabiFrq1,RabiFrq2,Tmax,PLOT_BOOL)
%T=0 to get steady state values. 
%PLOT_BOOL turns plotting on or off.
%|c>=|1>; |p>=|2>; |e>=|3>

%Might be interesting to modify this to see OBE in dressed state picture,
%e.g. do some sort of change of basis.

if ~exist('PLOT_BOOL','var')
    PLOT_BOOL = 1;
end

%run('constants.m')
hbar = 1.0546e-34;
epsilon0 = 8.8542e-12;
c = 299792458;


%coherences
gamma = 2*pi*6e6;
%coupling
Isat_1 = 7.59*10; %W/m^2
mu_13 = sqrt(c*epsilon0*gamma^2*hbar^2/(4*Isat_1));
E1 = hbar*RabiFrq1/mu_13;
Isat_2 = 7.59*10; %W/m^2
mu_23 = sqrt(c*epsilon0*gamma^2*hbar^2/(4*Isat_2));
E2 = hbar*RabiFrq2/mu_23;

%Hamiltonian
H = -hbar/2*[0, 0, -RabiFrq1;0,2*(Detune1-Detune2),-RabiFrq2;-conj(RabiFrq1),-conj(RabiFrq2),2*Detune1];
%G = [gamma11,gamma12,gamma13;gamma12,gamma22,gamma23;gamma13,gamma23,gamma33];
%anti-commutator with gamma is not complete. Leads to equations without
%proper# conservation. Fail to couple rho11 to rho22.

twoToone = @(ii,jj) (ii-1)*3+jj;
second = @(nn)mod(nn-1,3)+1;
first = @(nn)(nn-second(nn))/3+1;
oneTotwo = @(nn) [first(nn),second(nn)];
kDelta = @(aa,bb) aa==bb;

rhoRelationMat = zeros(9,9);
for nn=1:9
    for mm=1:9
        ind1 = oneTotwo(nn);
        ii = ind1(1);
        jj = ind1(2);
        ind2 = oneTotwo(mm);
        kk = ind2(1);
        ll = ind2(2);
        CoherentPart = kDelta(ll,jj)*H(ii,kk)-kDelta(kk,ii)*H(ll,jj);
        rhoRelationMat(nn,mm)=-1i/hbar*CoherentPart;
    end
end

decoherence = zeros(9,9);
decoherence(1,9) = gamma/2;
decoherence(5,9) = gamma/2;
decoherence(9,9) = -gamma;
decoherence(3,3) = -gamma/2;
decoherence(7,7) = -gamma/2;
decoherence(2,2) = -0/2;
decoherence(4,4) = -0/2;
decoherence(6,6) = -gamma/2;
decoherence(8,8) = -gamma/2;

rhoRelationMat = rhoRelationMat + decoherence;

if Tmax==0
    tol = 1e-5;
    [V,D]=eig(rhoRelationMat);
    lambdas = diag(D);
    [lambdas,I] = sort(lambdas);
    V = V(:,I);
    if abs(lambdas(1))<tol
        T=0;
        %normalize so Tr(rho)=1
        Y=V(:,1)'/(V(1,1)+V(5,1)+V(9,1));
        if abs(lambdas(2))<tol
            T=0;
            Y=zeros(1,9);
            disp('more than one zero eigenvalue')
        else
        end
    else
        T=0;
        Y=zeros(1,9);
        disp('Could not solve homogeneous equation')
    end
    
else
dy = @(t,y) rhoRelationMat*y;
%Tmax = 10*2*pi/min(omega1,omega2);
ts = [0, Tmax];
iconditions = [1,0,0,0,0,0,0,0,0];
[T,Y]=ode45(dy,ts,iconditions);
end

if PLOT_BOOL==1
    figure('name','3-Level OBEs Solution')
    for ii=1:9
        subplot(3,3,ii)
        if any([1,5,9]==ii)
            plot(T,Y(:,ii))
        else
            plot(T,abs(Y(:,ii)))
        end
    end
else
end
end
