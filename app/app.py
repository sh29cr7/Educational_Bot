import streamlit as st
import os
import json
import cv2

import logging
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


## Setting up langchain related stuff

from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
import boto3
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


    
#Create the connection to Bedrock
bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1', 
    
)

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    
)



bedrock_embeddings = BedrockEmbeddings(model_id = 'amazon.titan-embed-text-v1',)

langchain_llm_claude_sonnet = ChatBedrock(model_id = "anthropic.claude-3-sonnet-20240229-v1:0")
langchain_llm_meta_llama3_70b = ChatBedrock(model_id = "meta.llama3-70b-instruct-v1:0")



from pathlib import Path










# Setup logging
logger = logging.getLogger('rag')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


input_file_dir =f"../data/text_files" 
index_path     =f"../data/index"
import re

import re
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time 
import pandas as pd
def formattedtime(seconds):
    #print(f"formattedtime({seconds})")
    final_time = time.strftime("%H:%M:%S", time.gmtime(float(seconds)))
    return f"{final_time}"

def get_completion(messages):
    modelId = 'anthropic.claude-3-haiku-20240307-v1:0'

    converse_api_params = {
        "modelId": modelId,
        "messages": messages,
    }
    response = bedrock_runtime.converse(**converse_api_params)
    # Extract the generated text content from the response
    return response['output']['message']['content'][0]['text']

def remove_time_stamps(text):
    # Regular expression pattern to match time stamps in the format "00:24:40"
    pattern = r'\d{2}:\d{2}:\d{2}'
    
    # Remove time stamps using the regular expression substitution
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def post_process(text):

    soup =  BeautifulSoup(text,"html.parser") #parse html with BeautifulSoup
    file_name = soup.find('file_name').text

    start_time = soup.find('start_time').text #tag of interest <td>Example information</td>
    end_time = soup.find('end_time').text #tag of interest <td>Example information</td>
    
    answer = BeautifulSoup(text, "lxml").text
    answer = remove_time_stamps(answer)
    answer = answer.split("\n")
    return answer,start_time,end_time,file_name

prompt_template = """

Human: You are a educational tutorial AI system, and provides answers to questions by using fact based and statistical information when possible. 
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Also provide the start time, end time and the file name in the start_time, end_time and file_name tags respectively. 


<context>
{context}
</context>

<question>
{question}
</question>

Remember that you need to generate file name strictly as the actual file name mentioned in the context.
Don't Add any additional text like 'but no specific details about what a Decision Tree is are provided in the given context'
Assistant: Answer in <answer> tag followed by space('\n') and then start time, end time and file name in the <start_time>, <end_time> and <file_name> tags are as follows:"""





PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)






def main_page():
    st.markdown("# Educational Chatbot")
    query = st.text_input(label="Enter your Question!",)
    if st.button("Search"):

        vectorstore_chromaDB = Chroma(persist_directory=f"{index_path}/index_rag", embedding_function=bedrock_embeddings)

        qa = RetrievalQA.from_chain_type(
            llm=langchain_llm_claude_sonnet,
            chain_type="stuff",
            retriever=vectorstore_chromaDB.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa({"query": query})
        #st.write(f"unfiltered answer - {answer['result']}")
        
        ans,start_time,end_time,file_name = post_process(answer["result"])
        file = file_name
        st.write(f"-"*30)
        st.markdown("Claude Sonnet Result - ")
        st.write(f"\nAnswer:{ans}\n ")
        
              
        
        
        
        st.write(file_name)
        st.write(f"File Name : {file_name}")
        st.write(f"Start Time:{start_time}")
        st.write(f"End Time:{end_time}")      
             
        hr = start_time.split(":")[0]
        minute = start_time.split(":")[1]
        sec = start_time.split(":")[2]
        time_frames = int(hr)*60*60 + int(minute)*60 +  int(sec)
        
            
        qa = RetrievalQA.from_chain_type(
            llm=langchain_llm_meta_llama3_70b,
            chain_type="stuff",
            retriever=vectorstore_chromaDB.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa({"query": query})
        #st.write(f"unfiltered answer - {answer['result']}")

        ans,start_time,end_time,file_name = post_process(answer["result"])
        st.write(f"-"*30)
        st.markdown("Meta Llama3 70B Result - ")
        st.write(f"\nAnswer:{ans}\n ")
        
           

                

        st.write(f"File Name: {file_name}")
        st.write(f"Start Time:{start_time}")
        st.write(f"End Time:{end_time}")  
        
        
        with st.spinner("Generating multi-model output"):
            
            print("-"*30)
            print("Extracting the frames !")
            pathOut = f"../data/split_video"

            full_vid_path =f"../data/knowledgebase_video/{file}"
            os.system(f"rm -r {pathOut}")
            os.system(f"mkdir -p {pathOut}")
            # st.write(f"time_frame = {time_frames}")
            count = 0
            counter = 1

            vid = file.replace(" ","_")
            cap = cv2.VideoCapture(full_vid_path)
            st.write(full_vid_path)
            count = 0
            counter += 1
            success = True
            # st.write(f"time_frame = {time_frames}")
            while success:
                # st.write("yo")
                success,image = cap.read()
                # print('read a new frame:',success)
                if success == False:
                    break
                if count >= time_frames and count<=time_frames + 500:
                    # st.write("yo")
                    if count % 25 == 0:
                        #cv2.imwrite(pathOut + 'frame%d.jpg'%count,image)
                        print(f"writing -- {pathOut}/{vid}_{count}.jpg")
                        cv2.imwrite(f"{pathOut}/{vid}_{count}.jpg",image)       
                count+=1

            st.write("-"*30)
            st.write("Calling the llm !")


            image_dir = pathOut
            image_path_list = []
            answer_total = ""
            counter = 0
            prompt = f"""You are a question answering assistant. Answer the given question if its evidence 
            is present in the input image in the <answer> tab else just output <answer> don't no </output> without adding
            any additional text strictly. 
            <question>
            {query}
            </question>
            <instructions>
            - If answer is just not present just output <answer> don't know </answer> and don't add any additional 
            text strictly in this case since you will be penalised for additional text .
            - Strictly don't say things like 'I'm sorry, but I don't ' or 
            'I don't see any relevant information in the given images to answer this question.'
            'I do not have enough information to answer this question'
            etc. just say 'don't know'
            </instructions>
            """
            
            from tqdm import tqdm
            for images in tqdm(os.listdir(image_dir)):
                # print(images)
                if images == ".ipynb_checkpoints":
                    continue   
                with open(f"{image_dir}/{images}", "rb") as f:
                    img = f.read()

                content = [{"image": {"format": 'png', "source": {"bytes": img}}}]

                # Append the question to our images
                content.append({"text": prompt})

                messages = [
                    {
                        "role": 'user',
                        "content": content
                    },
                    {
                        "role": 'assistant',
                        "content": [{"text":"<answer>:"}]
                    }
                ]

                answer = get_completion(messages)



                answer_total = f"""{answer_total}\n\n 

                Answer :- {answer}"""

            answer_total = answer_total.replace("don't know","")
            answer_total = answer_total.replace("</answer>","")
            print(f"answer_total - {answer_total}")
            prompt = f"""The given context has answer tags since the output is generated from previous conversations.
            It may contain a lot of text saying answer is not present in the context and you need to ignore them.
            Answer might be present in some of the tags. Use them to answer the question. 
            Here is the  context:
            <context>
            {answer_total}
            </context>
            Answer the question 
            <question>
            {query}
            </question>

            """
            messages = [
                {
                    "role": 'user',
                    "content": [
                        {"text": prompt}

                    ]
                }
            ]
            answer = get_completion(messages)
            answer = BeautifulSoup(answer, "lxml").text
            st.markdown("Multi-modal output:")
            st.write(answer)
            
            image_dir = "../data/split_video"
            
            counter = 0
            for file in os.listdir(image_dir):
                counter = counter + 1
                if counter%5 == 0:
                    st.image(f"{image_dir}/{file}", caption=f"image-{counter}")

                



        
main_page()