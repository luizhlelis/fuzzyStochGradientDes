% Luiz Henrique Silva Lelis 2014034847

function lelisStochGradientDes()

    clear
    close all
    clc
    
    % n = numero de pontos de entrada - matriz x (itera sobre i)
    % m = numero de regras (itera sobre k)
    % d = numero de dimensoes - observacoes (itera sobre l)
    
    alpha = 0.1;                % Grau de aprendizado
    currentIteration = 1;       % Numero de iterações
    m = 100;                    % Numero de regras
    J(currentIteration) = 20;   % Custo inicial
    
    % Funcao sinc real (Usa 100 pontos de 0 a 2pi)
    x = linspace(0,2*pi,100)';
    [n,d] = size(x);
    yReal = sinc(x);
    
    % Pega os limites da matriz x
    minX = min(x);
    maxX = max(x);
    
    % Estimativa inicial para os parametros
    sig = rand(d,m);                 % Valores aleatorios entre 0 e 1
    c = (maxX-minX).*rand(d,m)+minX; % Valores aleatorios no intervalo dos pontos de entrada
    p = 2.*rand(d,m)-1;              % Valores aleatorios entre -1 e 1
    q = 2.*rand(1,m)-1;              % Valores aleatorios entre -1 e 1
    
    % Inicializa as matrizes das derivadas parciais e as demais
    dJdC = zeros(d,m);
    dJdSig = zeros(d,m);
    dJdP = zeros(d,m);
    dJdQ = zeros(d,m);
    mi = zeros(d,m);
    initialMi = zeros(d,m);
    mi1 = zeros(n,m);
    y11 = zeros(n,m);
    yHat = zeros(n,1);
    
    while J(currentIteration)>=1e-3
    
        % Itera sobre todos os pontos de entrada (calculo por amostra)
        for i=1:n

            % Funcoes de pertinencia calculadas para cada regra
            mi = pertinenceFunc(x, sig, c, i, m, d);
            
            % Armazena as fps iniciais para printar futuramente
            if (i == 1)
                initialMi = mi;
            end

            % Calcula o consequente da amostra atual
            y = p.*x(i) + q;

            % Calcula a saida do modelo (estimado) - w = mi -> uma dimensao
            yEst = sum(mi.*y)/(sum(mi));

            % Derivadas parciais calculadas para cada regra
            for k=1:m
                dJdC(k)   = partialJpartialC(yReal, yEst, x, y, sig, c, mi, i, k);
                dJdSig(k) = partialJpartialSig(yReal, yEst, x, y, c, sig, mi, i, k);
                dJdP(k)   = partialJpartialP(yReal, yEst, x, mi, i, k);
                dJdQ(k)   = partialJpartialQ(yReal, yEst, mi, i, k);
            end

            % Gradiente descendente - atualiza parametros
            c   = updateParameter(c, alpha, dJdC);
            sig = updateParameter(sig, alpha, dJdSig);
            p   = updateParameter(p, alpha, dJdP);
            q   = updateParameter(q, alpha, dJdQ);
        end

        % Calcula a fp e o consequente com os parametros atualizados
        for k=1:m
            mi1(:,k) = gaussmf(x, [sig(k) c(k)]);
            y11(:,k) = p(k)*x + q(k);
        end

        % Calcula a saida estimada com os parametros atualizados
        yHat = sum(mi1.*y11,2)./(sum(mi1,2));
        
        currentIteration = currentIteration + 1;

        J(currentIteration) = sum((yReal-yHat).^2);
        
        plotGraphs(x, yReal, yHat, y11, currentIteration);
        
    end
    
    plotCostGraph(J, currentIteration);
    
    plotPfGraph(x, initialMi, mi);
    
    RMSE = sqrt((1/length(x))*sum((sinc(x)-yHat).^2))

end

function mi = pertinenceFunc(x, sig, c, i, m, d)

    % Calcula a funcao de pertinencia
    mi = zeros(d,m);
    for k=1:m
        mi(k) = gaussmf(x(i), [sig(k) c(k)]);
    end
    
end

function dJdC = partialJpartialC(yReal, yEst, x, y, sig, c, mi, i, k)
    
    % Derivada parcial da função de custo em relacao ao paramentro c
    dJdC = -2*(yReal(i) - yEst)*(y(k)-yEst)/(sum(mi))*(mi(k))*(x(i)-c(k))/(sig(k)^2);

end

function dJdSig = partialJpartialSig(yReal, yEst, x, y, c, sig, mi, i, k)
    
    % Derivada parcial da função de custo em relacao ao paramentro sigma
    dJdSig = -2*(yReal(i)-yEst)*(y(k)-yEst)/(sum(mi))*(mi(k))*(x(i)-c(k))^2/(sig(k)^3);
    
end

function dJdP = partialJpartialP(yReal, yEst, x, mi, i, k)
    
    % Derivada parcial da função de custo em relacao ao paramentro p
    dJdP = -2*(yReal(i)-yEst)*(mi(k))/(sum(mi))*(x(i));
    
end

function dJdQ = partialJpartialQ(yReal, yEst, mi, i, k)
    
    % Derivada parcial da função de custo em relacao ao paramentro p
    dJdQ = -2*(yReal(i)-yEst)*(mi(k))/(sum(mi));
    
end

function newParam = updateParameter(param, alpha, partialDiff)

    % Estima os parametros do modelo utilizando as regras de 
    % atualizacao(gradiente descendente estocastico)
    
    newParam = param - alpha*partialDiff;
    
end

function plotGraphs(x, yReal, yHat, y11, currentIteration)
    
    % Plota os graficos
    figure(1)
    plot(x,yReal,'Linewidth',2);
    hold on;
    plot(x,yHat,'r','Linewidth',2);

    xlabel('x');
    ylabel('sinc(x)');
    legend('Funcao sinc(x) Real','Funcao sinc(x) Estimada');
    title('Funcao sinc(x) Real e Estimada');

    drawnow;
    hold off;
    
    % Salva os graficos de 5 em 5 iteracoes no formato eps
    if ((rem(currentIteration,5)==0)||(currentIteration==2))
        filename = strcat('resultados/iteracao',num2str(currentIteration));
        print(filename, '-depsc2');
        disp(['Iteracao -> ' num2str(currentIteration)]);
    end
    
    % Printa as retas
    %figure(4)
    %plot(x,y11);
    %xlabel('x');
    %ylabel('y_k');
    %title('Grafico das Retas y_k');
    
end

function plotCostGraph(J, currentIteration)
    
    % Grafico do custo J em funcao da epoca de treinamento
    figure(2)
    for idx=1:currentIteration
        plot(idx,J(idx),'or');
        hold on
    end
    
    axis([1 45 0 20])
    xlabel('Epocas');
    ylabel('Custo J');
    legend('Custo J');
    title('Grafico do custo J em função da epoca de treinamento');
    hold off
    
    % Salva o grafico no formato eps
    print('resultados/custo', '-depsc2');
    
end

function plotPfGraph(x, initialMi, mi)

    % Funcoes de pertinência iniciais (aleatorios) e apos a convergencia
    figure(3)
    plot(x,initialMi,'Linewidth',2);
    hold on;
    plot(x,mi,'r','Linewidth',2);

    xlabel('x');
    ylabel('gauss(x)');
    legend('FPs iniciais','FPs apos convergencia');
    title('Funcoes de pertinencia antes e depois da convergencia');

    drawnow;
    hold off;
    
    % Salva o grafico no formato eps
    print('resultados/FPs', '-depsc2');
    
end