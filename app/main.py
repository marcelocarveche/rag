# Importação de bibliotecas necessárias para o funcionamento do código
import os  # Permite interagir com o sistema operacional (acessar variáveis de ambiente, etc.)
import bs4  # Beautiful Soup 4: biblioteca para extrair dados de arquivos HTML e XML
from typing import List, Tuple  # Módulo para adicionar anotações de tipo (type hints)

# Importação de módulos específicos de várias bibliotecas
from dotenv import load_dotenv  # Carrega variáveis de ambiente de um arquivo .env
from langchain import hub  # Acessa prompts predefinidos do hub da LangChain
from langchain_chroma import Chroma  # Base de dados vetorial para armazenar embeddings
from langchain_community.document_loaders import WebBaseLoader  # Carrega documentos da web
from langchain_core.documents import Document  # Classe que representa documentos na LangChain
from langchain_core.output_parsers import StrOutputParser  # Converte saída do LLM para string
from langchain_core.runnables import RunnablePassthrough  # Permite passar dados entre componentes
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Interfaces para a API da OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divide texto em chunks

# Definição de valores padrão (constantes) para uso no programa
DEFAULT_MODEL = "gpt-4o"  # Modelo padrão da OpenAI a ser utilizado
DEFAULT_CHUNK_SIZE = 1000  # Tamanho padrão de cada pedaço (chunk) de texto em caracteres
DEFAULT_CHUNK_OVERLAP = 200  # Sobreposição padrão entre chunks adjacentes em caracteres
DEFAULT_PROMPT_TEMPLATE = "rlm/rag-prompt"  # Template de prompt padrão do hub da LangChain

def configurar_ambiente() -> None:
    """Carrega variáveis de ambiente e configura API keys.
    
    Esta função não retorna nada (None), apenas configura o ambiente de execução.
    """
    # Carrega variáveis do arquivo .env para as variáveis de ambiente do sistema
    load_dotenv()
    
    # Obtém a chave de API da OpenAI das variáveis de ambiente
    api_key = os.environ.get("OPENAI_API_KEY")  # Tenta obter a chave; retorna None se não existir
    
    # Verifica se a chave foi encontrada, caso contrário lança um erro
    if not api_key:
        # Levanta um erro que interrompe a execução do programa
        raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
    
    # Confirma a chave nas variáveis de ambiente (redundante, mas garante que está definida)
    os.environ["OPENAI_API_KEY"] = api_key

def carregar_documentos(urls: Tuple[str, ...]) -> List[Document]:
    """Carrega documentos a partir de URLs fornecidas.
    
    Args:
        urls: Tupla de URLs para carregar (imutável, por isso Tuple e não List).
        
    Returns:
        Lista de objetos Document contendo o conteúdo das páginas web.
    """
    try:
        # Cria um carregador de páginas web com configurações específicas
        loader = WebBaseLoader(
            web_paths=urls,  # URLs que serão carregadas
            bs_kwargs=dict(
                # Configura o Beautiful Soup para extrair apenas elementos com certas classes
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")  # Classes HTML a serem extraídas
                )
            ),
        )
        # Executa o carregamento e retorna os documentos
        return loader.load()
    except Exception as e:
        # Captura qualquer erro durante o carregamento
        print(f"Erro ao carregar documentos: {e}")  # Exibe mensagem de erro
        return []  # Retorna lista vazia em caso de erro

def dividir_documentos(documentos: List[Document], 
                       tamanho_chunk: int = DEFAULT_CHUNK_SIZE, 
                       sobreposicao: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """Divide documentos em chunks menores para processamento.
    
    Args:
        documentos: Lista de documentos para dividir.
        tamanho_chunk: Tamanho de cada chunk em caracteres (padrão=1000).
        sobreposicao: Quantidade de sobreposição entre chunks adjacentes (padrão=200).
        
    Returns:
        Lista de documentos divididos em pedaços menores.
    """
    # Cria um divisor de texto que quebra documentos recursivamente
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,  # Define o tamanho máximo de cada chunk
        chunk_overlap=sobreposicao  # Define a sobreposição entre chunks consecutivos
    )
    # Aplica a divisão e retorna os novos documentos (mais numerosos e menores)
    return text_splitter.split_documents(documentos)

def criar_vectorstore(documentos: List[Document]):
    """Cria uma base de dados vetorial a partir dos documentos.
    
    Args:
        documentos: Lista de documentos para armazenar como vetores.
        
    Returns:
        Objeto Chroma vectorstore com os documentos indexados.
    """
    try:
        # Cria e retorna uma base de dados Chroma com embeddings dos documentos
        return Chroma.from_documents(
            documents=documentos,  # Documentos a serem convertidos e armazenados
            embedding=OpenAIEmbeddings()  # Usa o modelo de embeddings da OpenAI
        )
    except Exception as e:
        # Captura qualquer erro durante a criação da base vetorial
        print(f"Erro ao criar vectorstore: {e}")
        raise  # Propaga o erro para ser tratado em um nível superior

def format_docs(docs):
    """Formata documentos para apresentação.
    
    Args:
        docs: Lista de documentos a serem formatados.
        
    Returns:
        String contendo o conteúdo de todos os documentos separados por linhas em branco.
    """
    # Junta o conteúdo de todos os documentos com dupla quebra de linha entre eles
    return "\n\n".join(doc.page_content for doc in docs)

def criar_rag_chain(retriever, modelo: str = DEFAULT_MODEL):
    """Cria uma cadeia de RAG (Retrieval Augmented Generation).
    
    Args:
        retriever: Retriever para buscar documentos relevantes.
        modelo: Nome do modelo LLM a ser usado.
        
    Returns:
        Cadeia de RAG configurada.
    """
    llm = ChatOpenAI(model=modelo)
    prompt = hub.pull(DEFAULT_PROMPT_TEMPLATE)
    
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    """Função principal que executa o fluxo RAG completo."""
    try:
        # Configurar ambiente
        configurar_ambiente()
        
        # Carregar e processar documentos
        urls = ("https://lilianweng.github.io/posts/2023-06-23-agent",)
        documentos = carregar_documentos(urls)
        
        if not documentos:
            print("Nenhum documento foi carregado. Verifique as URLs ou a conexão.")
            return
            
        chunks = dividir_documentos(documentos)
        
        # Criar vectorstore e retriever
        vectorstore = criar_vectorstore(chunks)
        retriever = vectorstore.as_retriever()
        
        # Criar e executar a cadeia RAG
        rag_chain = criar_rag_chain(retriever)
        
        # Exemplo de consulta
        resposta = rag_chain.invoke("O que é decomposição de tarefas?")
        print(resposta)
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")

if __name__ == "__main__":
    main()