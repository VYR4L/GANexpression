# GANexpression

Repositório para geração de imagens faciais para o Trabalho de Conclusão de Curso (TCC) sobre Reconhecimento de Expressões Faciais.

---

## Sobre o Projeto

Este projeto consiste no desenvolvimento de uma Rede Generativa Adversarial (GAN) voltada para a geração de imagens faciais com diferentes expressões emocionais.

A motivação surgiu diante da dificuldade de acesso ao dataset AffectNet, um dos maiores bancos de dados de expressões faciais, cujo acesso foi negado devido a restrições institucionais.

Para contornar essa limitação, o projeto utiliza um script autônomo baseado em **Playwright** para coletar imagens geradas sinteticamente pelo site [This Person Does Not Exist](https://thispersondoesnotexist.com/). Estas imagens são posteriormente transformadas pelo modelo GAN para representar diversas expressões emocionais.

Esta abordagem é especialmente importante, pois o dataset CK+ apresenta limitações, como distribuição não uniforme das emoções e imagens de baixa resolução, o que prejudica algumas aplicações.

---

## Funcionalidades

- Treinamento de GAN para geração de imagens faciais com diferentes expressões emocionais;
- Pipeline flexível para a inclusão e treinamento com diferentes datasets;
- Geração e salvamento automático de imagens ao longo do processo de treinamento;
- Configuração via linha de comando para facilitar experimentos.

---

## Próximos Passos

- Adaptação do modelo para geração de expreções faciais
