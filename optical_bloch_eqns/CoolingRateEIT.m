function [CoolingRate] = CoolingRateEIT(LambDickeParam,OmegaC,Delta,OmegaP,Gamma,OmegaTrap)
%C="Coupling", P="Probe". All values should be put in e.g.Gamma=(2*pi)*6e6
%Convention is blue detuning is positive Delta.
run('constants.m')

%Different than Morigi paper because she uses convention Delta<0 is
%blue-detuned.
Prefactor = 0.25*(OmegaC*OmegaP./sqrt(OmegaC.^2+OmegaP.^2)).^2;
Aplus = 0;
Aplus = Prefactor.*Gamma*OmegaTrap.^2./(((OmegaC.^2+OmegaP.^2)/4-OmegaTrap.*(OmegaTrap-Delta)).^2+Gamma^2*OmegaTrap.^2/4);
Aminus = Prefactor.*Gamma*OmegaTrap.^2./(((OmegaC.^2+OmegaP.^2)/4-OmegaTrap.*(OmegaTrap+Delta)).^2+Gamma^2*OmegaTrap.^2/4);

%return value which should be multiplied by hbar*OmegaTrap to get the
%heating rate. Returned value is thus in hbar*OmegaTrap/s
CoolingRate = LambDickeParam^2*(Aminus-Aplus);

end

