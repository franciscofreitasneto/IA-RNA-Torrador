# **Descrição do Projeto INOVAGRÃO - Software**
## Descrição do Sistema:
Para atender a necessidade de eficiência energética da torra de grãos de café de alta qualidade da região de Bom Jesus do Itabapoana e seu entorno, o Sistema deverá oferecer um suporte para a escolha de curvas de torra através do software "Artisan" (https://artisan-scope.org/).

## Objetivos do Sistema:
* Ser capaz de ler todos os lotes de torras fornecidos pelo "Artisan", extraindo as características de "temperatura dos grãos" e "tempo de torra";
* Além das informações fundamentais acima, o sitema deverá recolher dados do ambiente de torra e características específicas dos grãos, afim de detectar padrões que levem à sugestões de curvas específicas para cada caso;
* Para auxiliar a tomada de decisão, o sistema deve exportar a curva de torra no formato característico do "Artisan", permitindo ao operador importar os dados como uma refência a ser acompanhada na própria interface do "Artisan" durante a execução da torra.

## Requisitos do Sistema: Etapa 1
Para atender as características descritas acima, optou-se pelo teste de ferramentas de "Ciência de Dados" pela metodologia de prototipação afim de testar o sistema de suporte à decisão. As seguintes ferramentas foram selecionadas para testes: 
* **Carga e Manipulação da Base de Dados:**
  * Python;
  * Pandas;
  * Numpy;
* **Reconhecimento de Padrões e Inteligência Artifical:**
  * TensorFlow;
  * Keras;
