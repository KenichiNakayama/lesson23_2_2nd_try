"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st

# 条件付きインポート（エラー時はNoneを設定）
try:
    from docx import Document
except ImportError:
    Document = None

try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    WebBaseLoader = None
    CharacterTextSplitter = None
    OpenAIEmbeddings = None
    Chroma = None
    LANGCHAIN_AVAILABLE = False

import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成（環境変数チェック付き）
    try:
        initialize_retriever()
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.warning(f"Retriever initialization failed: {e}")
        # Retrieverなしでもアプリが動作するように空のRetrieverを設定
        st.session_state.retriever = None


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # LangChainが利用できない場合はスキップ
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain packages not available, skipping retriever initialization")
        st.session_state.retriever = None
        return
    
    # OpenAI APIキーの確認
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        st.session_state.retriever = None
        return
    
    # APIキーの基本的な形式チェック
    if not openai_api_key.startswith(("sk-", "sk-proj-")):
        logger.warning(f"Invalid OPENAI_API_KEY format: starts with {openai_api_key[:10]}...")
        st.session_state.retriever = None
        return
    
    logger.info(f"Using OpenAI API key starting with: {openai_api_key[:15]}...")
    
    # RAGの参照先となるデータソースの読み込み
    try:
        docs_all = load_data_sources()
        if not docs_all:
            logger.warning("No documents loaded for RAG")
            st.session_state.retriever = None
            return
        logger.info(f"Loaded {len(docs_all)} documents for RAG")
    except Exception as e:
        logger.error(f"Failed to load data sources: {e}")
        st.session_state.retriever = None
        return

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    try:
        embeddings = OpenAIEmbeddings()
        # 簡単なテストを実行してAPIキーが有効か確認
        test_embedding = embeddings.embed_query("test")
        logger.info("OpenAI embeddings initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embeddings: {e}")
        st.session_state.retriever = None
        return
    
    # チャンク分割用のオブジェクトを作成
    # 問題2: マジックナンバーを修正
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.RAG_CHUNK_SIZE,
        chunk_overlap=ct.RAG_CHUNK_OVERLAP,
        separator="\n"
    )

    # チャンク分割を実施
    splitted_docs = text_splitter.split_documents(docs_all)
    logger.info(f"Split documents into {len(splitted_docs)} chunks")

    # ベクターストアの作成
    try:
        db = Chroma.from_documents(splitted_docs, embedding=embeddings)
        logger.info("Vector store created successfully")
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        st.session_state.retriever = None
        return

    # ベクターストアを検索するRetrieverの作成
    # 問題1: kの値を3から5に変更
    # 問題2: マジックナンバーを修正
    try:
        st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RAG_SEARCH_K})
        logger.info(f"Retriever initialized successfully with k={ct.RAG_SEARCH_K}")
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        st.session_state.retriever = None
        return


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    # WebBaseLoaderが利用可能な場合のみWebページデータを読み込み
    if WebBaseLoader is not None:
        web_docs_all = []
        # ファイルとは別に、指定のWebページ内のデータも読み込み
        # 読み込み対象のWebページ一覧に対して処理
        for web_url in ct.WEB_URL_LOAD_TARGETS:
            try:
                # 指定のWebページを読み込み
                loader = WebBaseLoader(web_url)
                web_docs = loader.load()
                # for文の外のリストに読み込んだデータソースを追加
                web_docs_all.extend(web_docs)
            except Exception as e:
                logger = logging.getLogger(ct.LOGGER_NAME)
                logger.warning(f"Failed to load web content from {web_url}: {e}")
        # 通常読み込みのデータソースにWebページのデータを追加
        docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s