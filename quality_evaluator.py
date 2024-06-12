from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
from langchain.chains import RetrievalQA

class LLMQualityEvaluator:
    def __init__(self, model, vectorstore, chunks, num_examples=10):
        self.model = model
        self.vectorstore = vectorstore
        self.chunks = chunks[:num_examples]
        self.gen_chain = QAGenerateChain.from_llm(model)
        self.examples = []
        self.predictions = []

    def generate_examples(self):
        new_examples = self.gen_chain.apply_and_parse(
            [{"doc": t} for t in self.chunks]
        )
        for item in new_examples:
            self.examples.append(item['qa_pairs'])

    def setup_qa_chain(self):
        self.qa = RetrievalQA.from_chain_type(
            llm=self.model, 
            chain_type="stuff", 
            retriever=self.vectorstore.as_retriever(), 
            verbose=True,
            chain_type_kwargs={
                "document_separator": "<<<<>>>>>"
            }
        )

    def predict(self):
        if not self.examples:
            raise ValueError("No examples generated. Please run generate_examples() first.")
        self.predictions = self.qa.apply(self.examples)

    def evaluate(self):
        if not self.predictions:
            raise ValueError("No predictions made. Please run predict() first.")
        eval_chain = QAEvalChain.from_llm(self.model)
        graded_outputs = eval_chain.evaluate(self.examples, self.predictions)
        return graded_outputs