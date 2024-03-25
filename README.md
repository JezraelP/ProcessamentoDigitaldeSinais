# Cálculo Numérico da Transformada de Fourier
Para realizar o cálculo numérico da Transformada de Fourier, representada por G(f) , de uma função temporal g(t), é essencial utilizar amostras discretas de g(t). Devido à natureza discreta da computação, G(f) só pode ser calculada em frequências específicas e discretas, o que nos permite obter apenas amostras de G(f). Portanto, é necessário estabelecer relações entre as amostras de g(t) e as de G(f).

No contexto numérico, é imperativo trabalhar com dados limitados. Isso implica que tanto o conjunto de amostras de g(t) quanto o de G(f) devem ser finitos. Em outras palavras, lidamos com sinais que são restritos no tempo. Para sinais que originalmente não possuem essa limitação temporal, é preciso truncá-los, conferindo-lhes uma duração finita. Esta restrição se aplica igualmente a G(f).

Para um sinal discreto g(t), a Transformada de Fourier é dada por:
$$G(f) = \int_0^{T_0} g(t)e^{-j2\pi ft} dt \\
= \lim_{{T_s}\to0} \sum_{k=0}^{N_0-1} g(kT_s)e^{-j2\pi fk T_s}
$$

Em que $T_0$ é o um intervalo de tempo maior que a duração do sinal g(t), $T_s$ é o intervalo de amostragem do sinal, e $N_0$ é o número de amostras do sinal, e dado por: $N_0 = \frac{T_0}{T_s}$

Tomemos amostras de G(f) a intervalos uniformes de largura $f_0$. Seja $G_q$ a q-­ésima amostra, ou seja, $Gq = G(qf_0 )$; com isso, temos a transformada discreta:
$$G_q = \sum_{k_0}^{N_0-1} T_s g(kT_s)e^{-jq2\pi f_0 T_s k}$$
O valor de $T_s$ não pode tender a zero para praticidade computacional, logo, devemos escolhê-lo o menor possível do ponto de vista prático, o que causa um erro computacional.

A transformada inversa da DFT é dada por:
$$g_k = \frac{1}{N_0} \sum_{q = 0}^{N_0-1} G_q e^{jk\Omega_0 q}$$

Neste Repositório, usaremos MATLAB para calcular a DFT com o algoritmo da FFT.

As sequências $g_k$ e $G_q$ são ambas periódicas com um período de $N_0$. Isso significa que $g_k$ se repete a cada $T_0$ segundos, enquanto $G_q$ se repete a cada $f_s = \frac{1}{T_s}$ Hz, que é a frequência de amostragem. O intervalo entre as amostras de $g_k$ é $T_s$ segundos, e o intervalo de amostragem para $G_q$ é $f_0 = \frac{1}{T_0}$ Hz. O fato de g(t) ser limitado no tempo implica que sua transformada não é limitada em frequência, e como ela é uma função periódica, há sobreposição de componentes espectrais, causando erro. Ese erro é conhecido como **erro de mascaramento**(__aliasing__). O mascaramento pode ser reduzido com o aumento de $f_s$, e pode ser feito tão pequeno como desejado).

A sobreposição espectral pode ser vista como uma “dobra” do espectro que ocorre na frequência $f_s/2$, conhecida como **frequência de dobramento**. Se a frequência de dobramento for definida de tal forma que o espectro G(f) além dela seja insignificante, então a sobreposição espectral não terá impacto significativo. Assim, a frequência de dobramento deve ser, no mínimo, igual à maior frequência significativa do espectro, ou seja, a frequência após a qual G(f) se torna negligenciável. Esta frequência é denominada **largura de banda essencial** __B__ em hertz.
A quantidade de cálculos necessária para realizar uma Transformada Discreta de Fourier (DFT) foi significativamente diminuída graças a um algoritmo criado por Tukey e Cooley em 1965. Este algoritmo, chamado de **transformada de Fourier rápida** (FFT — __fast Fourier transform__), reduziu o número de cálculos de aproximadamente $N_0^2$ para $N_0 \log N_0$.

