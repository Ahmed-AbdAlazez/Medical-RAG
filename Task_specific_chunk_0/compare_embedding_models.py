from __future__ import annotations

import config
from embedding_models import JinaSmallEmbedding, TfidfEmbedding
from ingest import SimpleVectorIndex, chunk_documents, load_pdfs
import requests
import json
import config
import time




QUESTIONS = [
    "What recommendations did the committee make to ensure responsible use of ADHD medication?",
    "How can parents create better home and school environments for a child with ADHD?",
    "What discipline methods should parents learn for children with ADHD?",
]


def build_index_with_model(embedding_model):
    pages = load_pdfs(config.DATA_DIR)
    chunks = chunk_documents(pages)
    return SimpleVectorIndex(embedding_model, chunks)


def generate_rag_answer(question: str, search_results: list) -> dict:
    # 1. استخراج الـ Chunks والـ Citations من نتائج البحث
    retrieved_chunks = []
    citations = []
    context_text = ""
    
    for rank, (doc, score) in enumerate(search_results, 1):
        chunk_text = doc.page_content
        citation = doc.metadata.get('citation', 'Unknown')
        
        retrieved_chunks.append(chunk_text)
        citations.append(citation)
        
        # تجميع النص عشان نبعته للموديل
        context_text += f"--- Document {rank} (Source: {citation}) ---\n{chunk_text}\n\n"
        
    # 2. بناء الـ Prompt
    # 2. بناء الـ Prompt الصارم الخاص بالمساعد الطبي
    prompt = f"""You are a strict, precise, and professional medical AI assistant. Your ONLY job is to answer the user's question based strictly on the provided medical Context below.

    RULES:
    1. DO NOT use any external knowledge, personal knowledge, or information not explicitly stated in the Context.
    2. If the Context does not contain the answer, you MUST reply with exactly: "I don't have enough information to answer this based on the provided documents."
    3. Do not guess, infer, or make up any medical advice, diagnoses, or information.
    4. If the question is conversational or unrelated to the provided medical documents, simply reply that you are a medical assistant restricted to answering questions about the provided documents.

    Context:
    {context_text}

    Question: {question}

    Answer:"""

    # 3. إعداد الـ API Request لـ Groq مباشرة
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-oss-20b", # 👈 ده الموديل الجديد المتاح على Groq
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0 
    }
    # 4. إرسال الطلب واستقبال الإجابة
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result_json = response.json()
            llm_answer = result_json["choices"][0]["message"]["content"]
        else:
            llm_answer = f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        llm_answer = f"Connection Error: {str(e)}"
    
    # 5. إرجاع الـ 3 حاجات المطلوبة
    return {
        "answer": llm_answer,
        "chunks": retrieved_chunks,
        "citations": list(set(citations)) # set عشان نشيل أسماء الملفات المتكررة
    }


def answer_single_question(question: str, model_name: str, index: SimpleVectorIndex) -> None:
    print(f"\n🔍 Searching with {model_name}...")
    
    # 1. البحث وجلب الـ Chunks
    results = index.hybrid_search_with_relevance_scores(
        question,
        k=config.TOP_K,
        semantic_weight=config.HYBRID_SEMANTIC_WEIGHT,
        keyword_weight=config.HYBRID_KEYWORD_WEIGHT,
    )
    
    # 2. إرسال النتائج للـ LLM
    print("🤖 Thinking...")
    rag_output = generate_rag_answer(question, results)
    answer_text = rag_output['answer']
    
    # 3. طباعة الإجابة
    print(f"\n💡 Answer:\n{answer_text}")
    
    # === التعديل هنا: فحص الإجابة قبل طباعة المصادر ===
    # لو الموديل اعتذر عن الإجابة، هنطبع خط النهاية ونوقف (return)
    if "restricted to answering" in answer_text or "don't have enough information" in answer_text:
        print("-" * 50)
        return  # الكلمة دي بتخلي الكود يخرج من الدالة وميكملش طباعة اللي تحت
        
    # 4. طباعة المصادر والنصوص (هتطبع بس لو الموديل جاوب بجد)
    print(f"\n📚 Citations:\n{', '.join(rag_output['citations'])}")
    
    print("\n🧩 Chunks Used:")
    for i, chunk in enumerate(rag_output['chunks'], 1):
        snippet = " ".join(chunk[:150].split())
        print(f"  [{i}] {snippet}...")
    print("-" * 50)


def main() -> None:
    # 1. تعريف الموديل اللي هنستخدمه (خلينا نركز على Jina عشان أسرع وأدق)
    models = [
        (
            "JinaSmallEmbedding",
            JinaSmallEmbedding(
                model_name="jina-embeddings-v2-base-en",
                fallback_model=TfidfEmbedding(max_features=config.TFIDF_MAX_FEATURES),
                api_key=config.JINA_API_KEY # اتأكد إنك حاطط الـ API Key بتاعك هنا أو في الـ config
            ),
        )
    ]

    print("⏳ Building Vector Database... Please wait.")
    # 2. بناء الـ Index مرة واحدة في البداية
    indices = []
    for model_name, embedding_model in models:
        index = build_index_with_model(embedding_model)
        indices.append((model_name, index))
        
    print("✅ Database is ready!")
    print("=" * 50)
    print("👨‍⚕️ Welcome to the Medical AI Assistant!")
    print("Type your question below, or type 'exit' to close the program.")
    print("=" * 50)

    # 3. الـ Loop التفاعلي (الشات)
    while True:
        # هنا بياخد منك السؤال من الـ Terminal
        user_question = input("\n📝 Your Question: ")
        
        # لو كتبت exit أو quit بيقفل البرنامج
        if user_question.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
            
        # لو دوست Enter بالغلط من غير ما تكتب حاجة
        if not user_question.strip():
            continue

        # بيعدي على قاعدة البيانات ويجاوبك
        for model_name, idx in indices:
            answer_single_question(user_question, model_name, idx)
            time.sleep(2.5) # عشان الـ Rate Limit بتاع Groq

if __name__ == "__main__":
    main()