import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


#### memory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages.system import SystemMessage
from langchain_core.prompts.prompt import PromptTemplate

def get_memory(model):
    _DEFAULT_SUMMARIZER_TEMPLATE_KOR = """다음 example처럼 제공된 대화의 줄거리를 점진적으로 요약하고, 이전 요약에 추가하여 새로운 요약을 반환합니다.

    ### example start
    # 현재 요약:
    user는 AI가 인공 지능에 대해 어떻게 생각하는지 묻습니다. assitant는 인공 지능이 선한 힘이라고 생각합니다.

    # 새로운 대화:
    user: 인공 지능이 선한 힘이라고 생각하는 이유는 무엇입니까?
    assistant: 인공 지능이 인간이 잠재력을 최대한 발휘하는 데 도움이 될 것이기 때문입니다.

    # 새로운 요약:
    user는 AI가 인공 지능에 대해 어떻게 생각하는지 묻습니다. assistant는 인공 지능이 인간이 잠재력을 최대한 발휘하는 데 도움이 될 것이기 때문에 선한 힘이라고 생각합니다.
    ### example end

    # 현재 요약:
    {summary}

    # 새로운 대화:
    {new_lines}

    # 새로운 요약:"""
    SUMMARY_PROMPT_KOR = PromptTemplate(
        input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE_KOR
    )


    memory = ConversationSummaryBufferMemory(
        llm=model,
        max_token_limit=100,  # 요약의 기준이 되는 토큰 길이를 설정합니다.
        return_messages=True,
        prompt=SUMMARY_PROMPT_KOR
    )

    return memory

#### vectorstore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

PAPER_PROMPT = """논문의 요약 내용은 다음과 같습니다.
{paper_docs}
"""

RAG_PROMPT = """이 내용을 참고하여 답변해주세요.
{rag_docs}
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_splits(files):
    loader = PyPDFLoader(
        file_path= files,
    )
    docs = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500,   # 사이즈
        chunk_overlap=300, # 중첩 사이즈
    )

    split_docs = text_splitter.split_documents(docs)

    return split_docs

#### paper summary process
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

def summarize_paper(model, splits_docs):
    # old version
    map_template = """다음은 문서 중 일부 내용입니다
{pages}
이 문서 목록을 기반으로 주요 내용을 요약해 주세요.
답변:"""

    # Map 프롬프트
    map_prompt = ChatPromptTemplate([("human", map_template)])
    map_chain = LLMChain(llm=model, prompt=map_prompt)

    reduce_template = """다음은 요약의 집합입니다:
{doc_summaries}
이것으로 통합된 요약을 만들어 주세요.
답변:"""

    # Reduce 프롬프트
    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)


    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="doc_summaries" # Reduce 프롬프트에 대입되는 변수
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="pages",
        return_intermediate_steps=False,
    )

    rsts = map_reduce_chain.run(splits_docs)

    return rsts


#### model load
from dotenv import load_dotenv
import os

 # api key load
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

@st.cache_resource
def get_model():
    # gpt-4o-mini로 로컬 테스트... => hf model로 테스트 예정

    # chat gpt session open
    global api_key
    model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, max_completion_tokens=500, temperature=0.4)

    return model

####################################################################################
#### streamlit service
# session state 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = get_model()
if "memories" not in st.session_state:
    st.session_state.memories = get_memory(model=get_model())
if "papers" not in st.session_state:
    st.session_state.papers = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# 공간 분할, 논문 요약 보여주기
paper_summary = None
tmp_splits = None
with st.sidebar:
    st.subheader("paper summary")
    uploaded_files = st.file_uploader("choose a pdf file", type=['pdf'], accept_multiple_files=False)
    papers = st.session_state.papers.copy()
    if uploaded_files:
        # pdf 임시 저장
        temp_file = uploaded_files.name
        if(temp_file not in os.listdir()):
            with open(temp_file, "wb") as file:
                file.write(uploaded_files.getvalue())
                file_name = uploaded_files.name

        # state에 저장 유무 확인
        if uploaded_files.name in papers.keys():
            paper_summary = papers[uploaded_files.name]
        else:
            tmp_splits = get_splits(temp_file)
            st.session_state.vectorstore = Chroma.from_documents(
                    documents=tmp_splits,
                    embedding=OpenAIEmbeddings(api_key=api_key)
                )

            paper_summary = summarize_paper(model=st.session_state.model, splits_docs=tmp_splits)
            st.session_state.papers[uploaded_files.name] = paper_summary

        st.write(paper_summary)


st.title("Paper Summary Bot")
st.subheader("논문을 업로드 했다면, 논문에 관한 질문 시 꼭 '논문'을 붙여서 질문해주세요")
# 대화 이력
for content in st.session_state.messages:
    with st.chat_message(content["role"]):
        st.markdown(content['content'])

# 대화 시작
if prompt := st.chat_input(placeholder="say!"):
    # 요약 체크
    memory_rst = st.session_state.memories.load_memory_variables({})["history"]
    summary_content = []
    for his in memory_rst:
        if(isinstance(his, SystemMessage)):
            summary_content = [{"role":"user","type":"text", "text":his.content}]
            break

    # 질문
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # RAG
    rag_content = []
    paper_content = []
    if st.session_state.vectorstore:
        if "논문" in prompt:
            retriever = st.session_state.vectorstore.as_retriever()
            # 영문으로 변경
            tmp_message = HumanMessage(
                    content = [{"role":"user", "type":"text", "text":"'{}'를 영어로 번역해줘. 결과:".format(prompt)}]
                )
            tmp_result = st.session_state.model.invoke([tmp_message])
            eng_prompt = tmp_result.content.strip("")
            retrieved_docs = retriever.invoke(eng_prompt)
            rag_content = [{"role":"system", "type":"text", "text":RAG_PROMPT.format(rag_docs=format_docs(retrieved_docs))}]
            paper_content = [{"role":"system", "type":"text", "text":PAPER_PROMPT.format(paper_docs=paper_summary)}]


    # prompt
    content =  paper_content + rag_content + summary_content + [{"role": "user", "type": "text", "text":prompt}]

    message = HumanMessage(
        content = content
    )

    # gpt 응답
    result = st.session_state.model.invoke([message])
    response = result.content
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # 요약을 위한 대화 저장
    st.session_state.memories.save_context(
        inputs = {"user":prompt},
        outputs = {"assistant":response},
    )
