import base64
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages.system import SystemMessage

# memory 사전 작업
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

# api key load
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')


# chat gpt session open
model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, max_completion_tokens=500, temperature=0.4)

# streamlit service
# 제목
st.title("Playing With Image Bot")

# session state 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memories" not in st.session_state:
    st.session_state.memories = get_memory(model=model)

# 0 or None이면 False가 됨.
# file_uploader의 return은 None, UploadedFile, list of UploadedFile 세 가지로 이미지가 정상적으로 업로드 되면 True 판정
# 다중파일 옵션, accept_multiple_files = True 설정
if images:=st.file_uploader("사진을 자유롭게 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True):
    if(not isinstance(images, list)):
        images = [images] # 단일 파일의 경우 list 안에 담음

    base64_images = []
    for image in images:
        st.image(image) #streamlit에 이미지 표출
        base64_images.append(base64.b64encode(image.read()).decode("utf-8")) #gpt에 보내기 위해 base64로 인코딩

    # 이미지 content
    init_content = [{"role": "system", "type":"text", "text": "이미지와 관련된 대답을 해주세요."}]
    for im in base64_images:
        init_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{im}"}
                    })

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

        content = init_content + summary_content + [{"role": "user", "type": "text", "text":prompt}]

        message = HumanMessage(
            content = content
        )

        # gpt 응답
        result = model.invoke([message])
        response = result.content
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # 요약을 위한 대화 저장
        st.session_state.memories.save_context(
            inputs = {"user":prompt},
            outputs = {"assistant":response},
        )

