import logging
import sys
import os
import os.path as op
import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import CSVReader
from pathlib import Path
# https://sabeerali.medium.com/build-your-personal-rag-chatbot-chat-freely-with-your-data-powered-by-llamaindex-and-open-llms-63eb8ad1a053
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# specify path to CSV file, OPENAI api_key, and model below
FILE_PATH = "data"
assert op.exists(FILE_PATH), f"file not found at {FILE_PATH}, please check the file path."
# assert op.exists(
openai.api_key = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="Chatbot for doctor appointment", page_icon="🦙",
                   layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chatbot for doctor appointment")
st.info("แชทบอทช่วยตอบคำถามสำหรับการนัดหมายแพทย์ที่โรงพยาบาลศิริราช ปิยมหาราชการุณย์ ดูข้อมูลแพทย์เพิ่มเติมได้ที่ https://www.siphhospital.com/th/medical-services/find-doctor", icon="📃")

system_prompt = """
Given the following doctors' data in the file 'data' including output_clean.csv, output_fill_na.csv, output.md, sip_doctor_profiles.jsonl to create a response in Thai to a patient asking about scheduling an appointment,\
inquiring about the doctor's expertise, or seeking a recommendation for a doctor based on their needs. \
Note that user may inquire in a more casual text and you need to understand infer what they need before response.\
If user ask about doctor's data e.g. name, please provide information back in an easy to read format.\
Use only the data provided. The response should be in Thai and do not hallucinate. \
"""

system_prompt = """
<|SYSTEM|>#
Given a request or question in Prompt Answer.csv, generate a response based on a predefined set of examples. For each request, refer to a specific example that matches the context and content of the input. The examples consist of requests related to medical appointments and consultations,
along with appropriate responses specifying appointments with medical specialists. Use the examples to guide the generation of a relevant and specific response for each new request. \
Let's start by examining the contents of the uploaded file to understand its structure and the kind of information it contains. This will help us to craft a system prompt that allows the AI to effectively use the file as an example for generating every response. I'll read the file first. \
The uploaded file contains two columns: "Prompt" and "Answer." The "Prompt" column seems to include various requests or questions, primarily in Thai, related to medical appointments, consultations, or preparations for procedures. The "Answer" column provides responses to these prompts, specifying appointments with medical professionals in specific fields, also in Thai. \
To use this file as an example for generating every response, the system prompt should instruct the AI to interpret the "Prompt" column as input requests and generate responses based on the corresponding "Answer" column entries. Here's a proposed system prompt structure. \
"""

system_prompt = """
<|SYSTEM|>#
ถ้าถูกถามว่า ควรปรึกษาเเพทย์ผู้เชี่ยวช่าญหรือเเพทย์ที่ชำนาญท่านใด้ คำตอบของคำถามนี้ ควรบอกชื่อของเเพทย์, วันเเละเวลา, ด้านที่ชำนาญ ทั้งทุกองค์ประกอบให้ครบถ้วนเพื่อความชัดเจน \
คำตอบที่มี ชื่อ ตารางเวลา เเละ ด้านที่ชำนาญ
"""

system_prompt = """
<|SYSTEM|>#
ถ้าถูกถามว่า เกี่ยวกับ ด้านที่ชำนาญเเละเวลา ของเเพทย์ คำตอบของคำถามนี้ควร บอก วัน เวลา ตารางเวลา ของ เเพทย์ พร้อมด้วย ชื่อ เเละ ด้านที่ชำนาญอย่างชัดเจน เเละ ทั้งหมดต้องสอดคล้องกัน \
"""

system_prompt = """
<|SYSTEM|>#
ถ้าถูกถาม เกี่ยวกับเเพทที่มีความชำนาญในด้านใดด้านหนึ่ง เเต่ เเพทย์ที่ไม่มีตารางเวลา ควรตอบเเพทย์ที่มีตารางเวลาก่อน เเละ ถ้าไม่มีเเพทย์ที่มีตารางเวลาเลยในด้านความชำนาญที่ถูกถามให้ตอบว่าต้องติดต่อสอบถามกับทางโรงพยาบาลเนื่องจากเเพทย์มีตารางเวลาไม่เเน่ชัด \
"""

system_prompt = """
<|SYSTEM|>#
If the response is in English language, you have to translate them to thai before generate every response
and make sure all responses are in thai language. 
"""

@st.cache_resource(show_spinner=False)
def load_data(file_path: str):
    with st.spinner(text="Loading and indexing the Streamlit docs... hang tight! This should take 1-2 minutes."):
        # PandasCSVReader = download_loader("PandasCSVReader")
        # loader = PandasCSVReader()
        # docs = loader.load_data(file=Path(file_path))
        # # docs = SimpleDirectoryReader(file_path).load_data()
        # index = VectorStoreIndex.from_documents(
        #     docs, service_context=service_context)
        reader = SimpleDirectoryReader(input_dir=file_path, recursive=True)
        docs = reader.load_data()

        # data = CSVReader().load_data(file=Path(file_path))
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.01,
                     system_prompt=system_prompt)
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(
            docs, service_context=service_context)
    return index


index = load_data(FILE_PATH)
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
chat_engine = index.as_query_engine()


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "สอบถามข้อมูลการนัดหมายแพทย์ได้ที่นี่ครับ"}
    ]

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = chat_engine

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
# messages = getattr(st.session_state, 'messages', [])

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # response = st.session_state.chat_engine.chat(prompt)
            response = st.session_state.chat_engine.query(prompt)

            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            # Add response to message history
            st.session_state.messages.append(message)
