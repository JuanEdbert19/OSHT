from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy

class AImodel:

    def __init__ (self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        self.database = json.load(open("dataset.json"))
        self.chat_history = {}
        self.setup_vector()
        self.similarity_threshold = 0.25

    def setup_vector(self):
        self.entries = list(self.database.items())
        embeddings = self.encoder.encode([key for key, _ in self.entries])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(numpy.float32))

    def make_context(self, message: str, k=2):
        query_embedding = self.encoder.encode([message])
        distances, indices = self.index.search(query_embedding.astype(numpy.float32), k)

        max_similarity = 1 - distances[0][0]
        if max_similarity < self.similarity_threshold:
            return "" 

        context = "\n".join([self.entries[i][1] for i in indices[0]])
        return f"OSHIT Knowledge: {context}"
    

    async def make_response(self, user_id: int, message: str) -> str:

        if user_id not in self.chat_history:
            self.chat_history[user_id]= [
                {"role": "system",
        "content": """
            **You are the OSHIT Bot! You are expected to reply in a playful manner.** 
            Your personality:  
            1. Respond directly (No internal monologue
            2. Keep responses 2-4 sentences
            3. Never write "assistant" or role identifiers
            4. Use metaphors and jokes
            5. Use MAX 3 relevant emojis
            """}
            ]
        context = self.make_context(message)
        newmessage = f"{message}\n[CONTEXT: {context}]"
        self.chat_history[user_id].append({"role": "user", "content": newmessage})

    
        model_inputs = self.tokenizer.apply_chat_template(
                        self.chat_history[user_id], 
                        add_generation_prompt=True, 
                        return_tensors="pt", 
                        return_attention_mask=True)

        input_length = model_inputs.shape[1]

        generated_ids = self.model.generate(
                        model_inputs, 
                        do_sample=True,
                        max_new_tokens=150,
                        pad_token_id=self.tokenizer.eos_token_id
        )

        response=self.tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]

        self.chat_history[user_id].append({"role" : "assistant", "content":response})

        response = response.replace("CONTEXT:", "").replace("]", "").replace("]", "")
                      
        
        return response




