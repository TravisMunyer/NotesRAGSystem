from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Generator:
    def __init__(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def generate(self, context, query):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.llm(prompt)
