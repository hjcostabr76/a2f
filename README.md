# TODO

- Criar um arquivo 'main.py' (ou algum outro nome) para centralizar a execucao (pipeline);
- Esse arquivo poderia aceitar argumentos por linha de comando para parametrizar quais etapas do pipeline deverao ser executadas, por exemplo;

# Estrutura de pastas

```
file/[root_dir]
    |
    | -- blendshape [arquivos .json provenientes do a2f];
    |
    | -- wav [audios de entrada (formato .wav)];
    |
    | -- feature-lpc [representacao numerica de cada audio de entrada separado (gerado no pre-processamento)]
    |
    | -- model [modelos salvos (gerados pelo no treinamento)]
    |
    | -- dataset [features compiladas para treinamento (geradas no pre-processamento)]
	|
	| -- train
	| -- val

```
