units='si';
%SI constants. physics.nist.gov/cuu/Constants/index.html
K = 1.3806488e-23; % J/K
mu_bohr = 9.274009994e-24; % J/T
h = 6.62606957e-34; % J*s
hbar = h/(2*pi); % J*s
c = 2.99792458e8; % m/s
e = 1.602176565e-19; % C
mu0 = 4*pi*1e-7;
epsilon0 = 1/(c^2*mu0);


me=9.10938291e-31; % kg
mp=1.672621777e-27; % kg
alpha = 7.2973525698e-3; % dimensionless
abohr = hbar/(me*c*alpha); % m

% 6-lithium
mLi = 9.98834e-27; %Kg
%6Li D2 Line
lambda_D2=670.977e-9; %nm
omega_D2=2*pi*c/lambda_D2;
gamma_D2=2*pi*5.8724e6; %Hz
Isat_D2=25.4; %W/m^2

%6Li d1 line
lambda_D1=670.992421e-9; %nm
omega_D1 = 2*pi*c/lambda_D1;
gamma_D1 = 2*pi*5.8724e6; %Hz
Isat_D1 = 75.9; %W/m^2

m40K = 40*1.66e-27;
%trap params
Vlatt = 1.6e-3*K;
k=2*pi/(1064e-9);
omegaLatt = sqrt(Vlatt*k^2/mLi); %i.e. Sqrt[16*k^2*Vo/mLi], but Depth = Vlatt = 16Vo
omegaLS = 2*pi*163e3;
omega_Kuhr = 2*pi*300e3;
%Lamb-Dicke params
etaPlane = sqrt(hbar*(2*pi/671e-9)^2/(2*mLi)/omegaLatt);
etaZ = sqrt(hbar*(2*pi/671e-9)^2/(2*mLi)/omegaLS);
eta_Kuhr = sqrt(hbar*(2*pi/770.1e-9)^2/(2*m40K)/omega_Kuhr);

%Lithium and laser params
Gamma = 2*pi*6e6;
Isat = 7.59*10; %W/m^2
mu = sqrt(c*epsilon0*Gamma^2*hbar^2/(4*Isat));

%Laser coupling
Ic = 18*10*0.75; %EOM modulation removes ~25% of power.
Ec = sqrt(Ic/(epsilon0*c));
Ip = 18*10*0.125;
Ep = sqrt(Ip/(epsilon0*c));

%EIT parameters
Omegac = Gamma*sqrt(Ic/Isat/2)*sqrt(6);%need sqrt(# of beams) %mu*Ec/hbar;
Deltac = 2*pi*(228.2-2*87.3)*1e6;
Omegac_Kuhr = 2*pi*4.8e6; %2*pi*2.5e6; %kuhr # is all beams combined
Deltac_Kuhr = 10*Gamma;

Omegap = Gamma*sqrt(Ip/Isat/2)*sqrt(6);%mu*Ep/hbar;
Deltap = Deltac;
Omegap_Kuhr = 2*pi*1.6e6; %2*pi*1.4e6;
Deltap_Kuhr = 10*Gamma;

%EIT
GammaFano = Gamma*((Omegac/Deltac)^2+(Omegap/Deltap)^2);
ZeroToFano = (sqrt(Deltac^2+Omegac^2+Omegap^2)-abs(Deltac))/2;
MaxFanoScattering = omega_Kuhr*Gamma/4/sqrt(Deltac_Kuhr^2+Omegac_Kuhr^2+Omegap_Kuhr^2)/(2*pi);
GammaFano_Kuhr = Gamma*((Omegac_Kuhr/Deltac_Kuhr)^2+(Omegap_Kuhr/Deltap_Kuhr)^2)/(2*pi);
ZeroToFano_Kuhr =(sqrt(Deltac_Kuhr^2+Omegac_Kuhr^2+Omegap_Kuhr^2)-abs(Deltac_Kuhr))/2/(2*pi);


%limit where sitting exactly at the right place
0.5*omega_Kuhr*Gamma/(4*sqrt(Deltap_Kuhr^2+Omegap_Kuhr^2+Omegac_Kuhr^2)); %Quantu per ms

%change from paper to use convention Delta>0 is blue detuning.
Aplus = @(w1,w2,D,wlatt,Gam)0.25*(w1*w2/sqrt(w1^2+w2^2))^2*Gam*wlatt^2/(((w1^2+w2^2)/4-wlatt*(wlatt-D))^2+Gam^2*wlatt^2/4);
Aminus = @(w1,w2,D,wlatt,Gam)0.25*(w1*w2/sqrt(w1^2+w2^2))^2*Gam*wlatt^2/(((w1^2+w2^2)/4-wlatt*(wlatt+D))^2+Gam^2*wlatt^2/4);
CRate = @(eta,w1,w2,D,wlatt,Gam)eta^2*(Aminus(w1,w2,D,wlatt,Gam)-Aplus(w1,w2,D,wlatt,Gam));
%may need to be more careful. Cooling only happens due to one direction,
%but Light Shift from all directions.
CRate_Kuhr = CRate(2*eta_Kuhr,Omegac_Kuhr,Omegap_Kuhr,Deltac_Kuhr,omega_Kuhr,Gamma);



PLOTTING=1;
if PLOTTING==1
figure
[Dp_Kuhr,chi_Kuhr] = SusceptibilityEIT(Omegap_Kuhr,Deltac_Kuhr,Omegac_Kuhr,Gamma,1);
subplot(2,1,1)
title('EIT Spectrum, Kuhr Group Values')
subplot(2,1,2)
hold on;
plot([10+omega_Kuhr/Gamma,10+omega_Kuhr/Gamma],[0,1],'b')
plot([10-omega_Kuhr/Gamma,10-omega_Kuhr/Gamma],[0,1],'b')


%5mm waist @8mw
Itot = 180;
wc = Gamma*sqrt(Itot*0.75/Isat/2)*sqrt(6);
wp = Gamma*sqrt(Itot*0.125/Isat/2)*sqrt(6);
[Dc_lowerI,Chi_lowerI]=SusceptibilityEIT(wp,Deltac,wc,Gamma,0);
%3750um waist @8mw.
%750um wasit @600uw
Itot = 362;
wc = Gamma*sqrt(Itot*0.75/Isat/2)*sqrt(6);
wp = Gamma*sqrt(Itot*0.125/Isat/2)*sqrt(6);

[Dc_higherI,Chi_higherI]=SusceptibilityEIT(wp,Deltac,wc,Gamma,0);

figure
hold all;
plot(Dc_lowerI/Gamma,imag(Chi_lowerI)/max(abs(imag(Chi_lowerI))),'b.')
plot(Dc_higherI/Gamma,imag(Chi_higherI)/max(abs(imag(Chi_higherI))),'g.')
plot([Deltac/Gamma+omegaLatt/Gamma,Deltac/Gamma+omegaLatt/Gamma],[0,1],'b')
plot([Deltac/Gamma+omegaLS/Gamma,Deltac/Gamma+omegaLS/Gamma],[0,1],'c')
legend('2.5 Isat','5 Isat','wLatt','wLS')
plot([Deltac/Gamma-omegaLatt/Gamma,Deltac/Gamma-omegaLatt/Gamma],[0,1],'b')
plot([Deltac/Gamma-omegaLS/Gamma,Deltac/Gamma-omegaLS/Gamma],[0,1],'c')





else
end
