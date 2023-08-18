if __name__=="__main__":
    from haystack.document_stores import InMemoryDocumentStore
    document_store = InMemoryDocumentStore(use_bm25=True)
    from haystack.nodes import PDFToTextConverter
    pdf_converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
    converted = pdf_converter.convert(file_path = "docs\*.pdf", meta = { "company": "Company_1", "processed": False })
    document_store.write_documents(converted)
    from haystack.nodes import BM25Retriever
    retriever = BM25Retriever(document_store=document_store, top_k=2)
    from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
    rag_prompt = PromptTemplate(
        prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             Your answer should be in your own words and be no longer than 50 words.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
        output_parser=AnswerParser(),   
    )

    prompt_node = PromptNode(model_name_or_path="google/flan-t5-large", default_prompt_template=rag_prompt)
    from haystack.pipelines import Pipeline

    pipe = Pipeline()
    import time
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])
    start_time = time.time()
    output = pipe.run(query=input())

    print(output["answers"][0].answer)
    print("--- %s seconds ---" % (time.time() - start_time))
