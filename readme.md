# Projeto RAG (Retrieval-Augmented Generation)

## O que é RAG?

RAG (Retrieval-Augmented Generation) é uma técnica que combina sistemas de recuperação de informações com modelos de linguagem generativa para produzir respostas mais precisas e fundamentadas.

O processo do RAG funciona da seguinte forma:

1. **Recuperação (Retrieval)**: Busca informações relevantes em uma base de conhecimento externa
2. **Aumento (Augmentation)**: Enriquece o contexto do modelo com as informações recuperadas
3. **Geração (Generation)**: Produz respostas baseadas tanto no conhecimento do modelo quanto nas informações recuperadas

As vantagens do RAG incluem:

- Respostas mais precisas e atualizadas
- Redução de alucinações (informações inventadas pelo modelo)
- Capacidade de citar fontes específicas
- Maior transparência no processo de geração de respostas

## Sobre este Projeto

Este projeto implementa um sistema RAG completo usando a biblioteca LangChain e a API da OpenAI. O sistema é capaz de:

1. Carregar documentos de URLs da web
2. Processar e dividir o texto em chunks gerenciáveis
3. Criar embeddings vetoriais dos documentos
4. Armazenar os embeddings em uma base de dados vetorial (Chroma)
5. Recuperar informações relevantes com base em consultas
6. Gerar respostas fundamentadas usando um modelo de linguagem

### Fluxo de Funcionamento

O script `main.py` implementa o seguinte fluxo de trabalho:

1. **Configuração do Ambiente**: Carrega variáveis de ambiente e configura a API key da OpenAI
2. **Carregamento de Documentos**: Extrai conteúdo de páginas web especificadas
3. **Processamento de Documentos**: Divide os documentos em chunks menores para melhor processamento
4. **Indexação Vetorial**: Cria uma base de dados vetorial para armazenar e buscar documentos
5. **Criação da Cadeia RAG**: Configura um pipeline que combina recuperação e geração
6. **Consulta e Resposta**: Executa consultas e gera respostas fundamentadas nos documentos recuperados

### Componentes Principais

- **WebBaseLoader**: Carrega conteúdo de páginas web
- **RecursiveCharacterTextSplitter**: Divide textos em chunks menores
- **OpenAIEmbeddings**: Cria representações vetoriais dos textos
- **Chroma**: Base de dados vetorial para armazenar e buscar documentos
- **ChatOpenAI**: Interface com o modelo de linguagem da OpenAI
- **RAG Chain**: Pipeline que combina os componentes para recuperação e geração

## Como Usar

1. Clone este repositório
2. Crie um arquivo `.env` na raiz do projeto com sua chave da OpenAI:
