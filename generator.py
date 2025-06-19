from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Generator:
    def __init__(self, model_name, max_tokens=1000):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_tokens)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def generate(self, context, query):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        generated_text = self.llm(prompt)
        answer = generated_text.split("Answer:")[-1].strip()
        return {"answer": answer}
