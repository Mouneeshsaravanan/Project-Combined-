from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.api_core.exceptions as google_exceptions
from huggingface_hub import InferenceClient

def get_llm(gemini_api_key, hf_api_key):
    if not gemini_api_key or gemini_api_key.strip() == "":
        print("Warning: No valid Gemini API key provided. Attempting fallback to Hugging Face.")
        if hf_api_key:
            return InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_api_key), "huggingface"
        else:
            raise ValueError("No Gemini API key provided and no Hugging Face API key available. Please set a valid API key.")

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
        llm.invoke("Test")
        print("Successfully initialized Gemini LLM.")
        return llm, "gemini"
    except google_exceptions.InvalidArgument as e:
        print(f"Gemini API key or model error: {str(e)}")
        if hf_api_key:
            print("Falling back to Hugging Face due to Gemini API key or model issue.")
            return InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_api_key), "huggingface"
        else:
            raise ValueError(f"Invalid Gemini API key or unsupported model: {str(e)}. No Hugging Face API key provided for fallback.")
    except google_exceptions.ResourceExhausted as e:
        print(f"Gemini API rate limit exceeded: {str(e)}")
        if hf_api_key:
            print("Falling back to Hugging Face due to Gemini rate limit.")
            return InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_api_key), "huggingface"
        else:
            raise ValueError("Gemini API rate limit exceeded and no Hugging Face API key provided.")
    except Exception as e:
        print(f"Unexpected error initializing Gemini LLM: {str(e)}")
        if hf_api_key:
            print("Falling back to Hugging Face due to unexpected Gemini error.")
            return InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=hf_api_key), "huggingface"
        else:
            raise ValueError(f"Error initializing LLM: {str(e)}. No Hugging Face API key provided.")

def generate_rag_response(llm, llm_type, query, relevant_docs, chat_history):
    context = "\n".join([doc.page_content for doc in relevant_docs])
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[:-1]])
    
    prompt = f"""Chat history:
{history_str}

Context from documents:
{context}

User query: {query}

Answer based on the context and history:"""
    
    if llm_type == "gemini":
        chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["query"], template="{query}"))
        response = chain.run(query=prompt)
    else:
        response = llm.text_generation(prompt, max_new_tokens=500, temperature=0.7)
    
    return response

def generate_pure_llm_response(llm, llm_type, query, chat_history):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[:-1]])
    
    prompt = f"""Chat history:
{history_str}

User query: {query}

Answer:"""
    
    if llm_type == "gemini":
        chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["query"], template="{query}"))
        response = chain.run(query=prompt)
    else:
        response = llm.text_generation(prompt, max_new_tokens=500, temperature=0.7)
    
    return response