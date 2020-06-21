LineWidth = 2*pi*6e6;
RabiFrq = 2*pi*2.5e6;
Detunings = -5*LineWidth:LineWidth/10:5*LineWidth;

RhoVals = zeros(4,length(Detunings));
for ii=1:length(Detunings)
    [RhoVals(:,ii),~] = OBE2([Detunings(ii),RabiFrq,LineWidth],0,0); 
end

%instead of plotting off diagonal elements, plot their real and imaginary
%parts. i.e. the real and imaginary parts of susceptibility.
figure('name','Lineshape From Two-Level OBEs')
subplot(2,2,1)
plot(Detunings/LineWidth,RhoVals(1,:))
xlabel('Detuning (\Gamma)')
grid on;
title('\rho_{gg}')

subplot(2,2,2)
plot(Detunings/LineWidth,real(RhoVals(3,:)))
grid on;
title('\rho_{ge}')

subplot(2,2,3)
plot(Detunings/LineWidth,imag(RhoVals(4,:)))
grid on;
title('\rho_{eg}')

subplot(2,2,4)
plot(Detunings/LineWidth,RhoVals(2,:))
grid on
title('\rho_{ee}')

sgtitle('Lineshape From Two-Level OBEs');