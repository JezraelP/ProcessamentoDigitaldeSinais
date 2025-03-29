% Cálculo da DFT por FFT

%Para um sinal x, é necessário a priori determinar os seguintes parâmetros:
% T0: Período de observação do sinal
%Ts: Período de Amostragem

%Para o sinal e^-2t * u(t), a transformada de Fourier é 1/(j2πf + 2)
%Essa função não é limitada em frequência, logo, é necessário truncá-la.

%Escolhemos para a banda essencial o valor para qual o ódulo é igual a 1% do valor máximo
%O módulo dessa função é dado por:
pkg load symbolic
syms f
F = abs(1/(2*j*pi*f + 2)); %Módulo do espectro do sinal

%Note que o módulo da função é dado por: mod = 1/(sqrt((2*pi*f)**2 + 4))
%Assim, seu valor máximo ocorre em f = 0, e é 0.05
max = 0.5
modB = max*0.01
eq = 1 / (sqrt((2 * pi * f) ^ 2 + 4)) == modB; % Define a equação para achar a frequência essencial
B = solve(eq, f); % Resolve a equação para 'f'

%Como Ts precisa ser menor ou igual a 1/2B,
Tsmax = double(1/(2*B(2)));
Ts = 1/64;

Ts<Tsmax;

%Para escolher T0, já que o sinal não é limitado no tempo, escolhemos um valor tal que g(t0) << 1.
%Fazendo T0 = 4, obtemos um valor de oito constante sde tempo do sinal exponencial.

%É interessante que, para o cáclculo da FFT de um sinal, o valor de N seja uma potência de 2
T0 = 4;
N = T0/Ts;

%Agora o algoritmo que calcula a FFT do sinal e^-2t * u(t):

t = 0:Ts:Ts*(N-1), t = t';
g = Ts*exp(-2*t);
g(1)=Ts*0.5;
G=fft(g);
[Gp,Gm]=cart2pol(real(G), imag(G));
k=0:N-1; k=k';
w=2*pi*k/T0;

subplot(211), stem(w(1:32), Gm(1:32));
title('Espectro de Magnitude');
xlabel('Frequência (rad/s)');
ylabel('|G(f)|');

subplot(212), stem(w(1:32), Gp(1:32));
title('Espectro de Fase');
xlabel('Frequência (rad/s)');
ylabel('Fase (radianos)');
% Nesse caso, conhecemos a expressão de G(f). Normalmente isso não ocorre
%NEntão deve-se abaixar o valor de Ts até que o resultado seja satisfatório

