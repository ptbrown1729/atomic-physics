function [delta_probes,chis]=SusceptibilityEIT(omegap,deltac,omegac,gamma,PLOT_BOOL)
% Get susceptability. First transform rho to non-rotating basis,
%take Tr{rho*mu) and match with susceptibility terms.

% gamma = 2*pi*6e6;
% omegac = 2*pi*2.5e6;
% omegap = 2*pi*1.4e6;
% deltac = 10*gamma;
deltapMin = min(-20*gamma,deltac-20*gamma);
deltapMax = max(20*gamma,deltac+20*gamma);
deltap = deltapMin:gamma/5:deltapMax;
%deltac-20*gamma:gamma/5:deltac+20*gamma; 

N = 1;
chi = zeros(1,length(deltap));
for ii=1:length(deltap)
    DisplayPlots = 0;
    [~,Y] = ThreeLevelOBE(deltap(ii),deltac,omegap,omegac,0,DisplayPlots);
    rho_pe_temp = Y(end,3);
    if imag(rho_pe_temp)<0
        %correction since eigenvector has sign freedom
        rho_pe_temp = -rho_pe_temp;
    end
    %chi(ii) = 2*N*abs(mu_pe)*rhope/(Ep*epsilon0);
    chi(ii) = rho_pe_temp;
end

NChiReal = max(abs(real(chi)));
chiRealNorm = real(chi)/NChiReal;
NChiImag = max(abs(imag(chi)));
chiImagNorm = imag(chi)/NChiImag;

%do some processing to ensure we have good enough resolution everywhere.
FinerGrain = ones(1,length(chi)-1);
maxIterations = 10;
iterator = 0;
while (max(FinerGrain)&&(iterator<maxIterations))~=0
    tol = 7e-2;
    chiRealDifferences = abs(chiRealNorm(2:end)-chiRealNorm(1:end-1));
    chiImagDifferences = abs(chiImagNorm(2:end)-chiImagNorm(1:end-1));
    FinerGrain=max(chiRealDifferences>tol,chiImagDifferences>tol);
    FinerDeltap = [];
    for ii=1:(length(chi)-1)
        if FinerGrain(ii)==1
            del = (deltap(ii+1)-deltap(ii))/10;
            FinerMesh = (deltap(ii))+del:del:(deltap(ii+1)-del);
            FinerDeltap = cat(2,FinerDeltap,FinerMesh);
        else
        end
    end
    [deltap,I] = sort(cat(2,deltap,FinerDeltap));
    
    chiFiner = [];
    for ii=1:length(FinerDeltap)
        [~,Y] = ThreeLevelOBE(FinerDeltap(ii),deltac,omegap,omegac,0,DisplayPlots);
        rho_pe_temp = Y(end,3);
        if imag(rho_pe_temp)<0
            %correction since eigenvector has U(1) freedom.
            rho_pe_temp = -rho_pe_temp;
        end
        chiFiner = cat(2,chiFiner,rho_pe_temp);
    end
    chi = cat(2,chi,chiFiner);
    chi = chi(I);
    chiRealNorm = cat(2,chiRealNorm,real(chiFiner)/NChiReal);
    chiRealNorm = chiRealNorm(I);
    chiImagNorm = cat(2,chiImagNorm,imag(chiFiner)/NChiImag);
    chiImagNorm = chiImagNorm(I);
    iterator = iterator+1;
    disp('iteration complete')
end

chis = chi;
delta_probes = deltap;

if PLOT_BOOL==1
    subplot(2,1,1)
    plot(deltap/gamma,chiRealNorm,'b.')
    hold on;
    subplot(2,1,2)
    plot(deltap/gamma,chiImagNorm,'b.')
    hold on;
    plot([deltap(1)/gamma,deltap(end)/gamma],[0,0],'r')
    hold on;
else
end
end


