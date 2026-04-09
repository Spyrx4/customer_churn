from rag_chatbot import search, generate_answer
from prompts import SYSTEM_PROMPT
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

history = [{
    'role': 'system',
    'content': SYSTEM_PROMPT
}]

while True:
    raw_query = input(f'You: \n').strip()
    
    openai_client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    prompt_enhancement = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role':'system', 'content':'You are a user question translator. Translate user questions from any language into Indonesian and make them clearer and more detailed.'},
            {'role':'user', 'content':raw_query},
        ],
        max_tokens=150
    )
    
    query = prompt_enhancement.choices[0].message.content
    
    if not query:
        continue
    
    results = search(query, n_result=3)
    context = "\n".join([chunk['text'] for chunk in results])
    
    user_prompt = f"""Customer Question : {query}
    
    Context from Nusantara Connect:
    {context}
    """
    
    history.append({'role': 'user',
                    'content': user_prompt})
    
    answer = generate_answer(history)
    
    
    history[-1] = {'role': 'user',
                   'content': query}
    
    print(f'Rini: {answer}\n')