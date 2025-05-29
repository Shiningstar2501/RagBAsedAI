import pandas as pd

df = pd.read_excel("about_us_info.xlsx")

data= df["Extracteed Data"].astype(str).tolist()

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

query = "What is the address in this text?"
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding), k=5)  # get top 5 similar rows

relevant_rows = [data[i] for i in I[0]]

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# llm = OpenAI(temperature=0)
llm = Ollama(model="llama3")

template = """You are an AI assistant. Extract only the address from the following text:
Text: {text}
Address:"""

prompt = PromptTemplate(input_variables=["text"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

for row in relevant_rows:
    print(chain.run(row))
