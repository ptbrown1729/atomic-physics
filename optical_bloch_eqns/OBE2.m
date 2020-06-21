function [RhoMatrixElems,T] = OBE2(P,Tmax,PLOT_BOOL)
%function [RhoMatrixElems,T] = OBE2(P,Tmax,PLOT_BOOL)
%Solves the two-level Optical Bloch Equations
%P = [Detuning, RabiFrq, LineWidth]. Where Detuning = 2pi#.
%Set Tmax = 0 to solve steady state problem. 
%PLOT_BOOL turns plotting on or off.
%Uses initial condition rho_gg = 1.
%Rho Vector [rho_gg,rho_ee,rho_ge,rho_eg]

if ~exist('PLOT_BOOL','var')
    PLOT_BOOL = 1;
end

Detuning = P(1);
RabiFrq = P(2);
LineWidth = P(3);
%solves in rotating frame
FirstR = [0, LineWidth, -1i/2*RabiFrq, 1i/2*conj(RabiFrq)];
SecondR = [0 -LineWidth, 1i/2*RabiFrq, -1i/2*conj(RabiFrq)];
ThirdR = [-1i/2*conj(RabiFrq), 1i/2*conj(RabiFrq), -(LineWidth/2+1i*Detuning),0];
FourthR = [1i/2*RabiFrq, -1i/2*RabiFrq, 0, -(LineWidth/2-1i*Detuning)];
OpticalBEmat = cat(1, FirstR, SecondR, ThirdR, FourthR);

%Solve Steady State
if Tmax == 0
     tol = 1e-5;
    [V, D]=eig(OpticalBEmat);
    lambdas = diag(D);
    [lambdas, I] = sort(lambdas);
    V = V(:,I);
    if abs(lambdas(1))<tol
        T=0;
        %V(:,1) gives steady state solution to OBE. Normalize so Tr(rho)=1.
        RhoMatrixElems=V(:,1)'/(V(1,1)+V(2,1));
        if abs(lambdas(2))<tol
            T=0;
            RhoMatrixElems=zeros(1,4);
            disp('more than one zero eigenvalue')
        else
        end
    else
        T=0;
        RhoMatrixElems=zeros(1,4);
        disp('Could not solve homogeneous equation')
    end
%Solve time dependence
else
dy = @(t,y) OpticalBEmat*y;
iconditions = [1,0,0,0];
[T, RhoMatrixElems] = ode45(dy, [0,Tmax], iconditions);
end

if PLOT_BOOL == 1
    %Plot matrix elements. Position on plot grid mirrors position in
    %matrix.
    FigName = '2-Level OBEs Solution';
    TitleString = sprintf('RabiFrq=%.2f, Detuning=%.2f',RabiFrq/LineWidth,Detuning/LineWidth);
    
    fighandle = figure('name',FigName);
    subplot(2,2,1)
    plot(T,RhoMatrixElems(:,1))
    grid on;
    title(TitleString);
    
    subplot(2,2,4)
    plot(T,RhoMatrixElems(:,2))
    grid on;
    title('\rho_{ee}')
    
    subplot(2,2,2)
    plot(T,abs(RhoMatrixElems(:,3)))
    grid on;
    title('\rho_{ge}');
    
    subplot(2,2,3);
    plot(T,abs(RhoMatrixElems(:,4)));
    grid on;
    title('\rho_{eg}');
    
else
end
end
