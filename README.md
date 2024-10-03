# RAG Evaluation Using Bedrock

This repository demonstrates how to evaluate the quality of a RAG (Retrieval-Augmented Generation) pipeline using the Bedrock API.

## 01-test-dataset-generation.ipynb

To evaluate RAG quality, we need to create an evaluation dataset. While it's possible to create datasets manually, it's time-consuming. Therefore, using LLMs to generate synthetic datasets is a popular approach. Here's how we generate the desired number of synthetic Q&A pairs:

1. Load a PDF document
2. Split the document into chunks of a specified length
3. Randomly select a chunk and generate a complex/simple question and its corresponding answer (ground truth) based on that chunk
4. Repeat the chunk selection and Q&A generation process N times

The completed test dataset is structured as follows:
```
{
    "quesiton": "{generated question}",
    "ground_truth": "{generated answer}",
    "question_type": "complex | simple",
    "contexts": "{randomly selected context used for Q&A generation}
}
```

## 02-ragas-evaluation.ipynb

This notebook uses the test dataset to evaluate RAG quality. It supports evaluation of RAGAS metrics including AnswerRelevancy, Faithfulness, ContextRecall, and ContextPrecision. 

While relying on existing RAGAS evaluation metrics and calculation methods, the following modifications have been made:

- Uses Bedrock's converse API instead of LangChain or LlamaIndex frameworks
- Employs tool use prompting techniques to ensure accurate JSON parsing of LLM outputs
- Changes asynchronous LLM calls to synchronous to avoid API throttling when dealing with large Q&A test sets
- Modifies segment separation for Faithfulness and Recall calculations from sentence-based (`.endswith(".")`) to paragraph-based (`'\n\n'`) for better context continuity
- Simplifies the Faithfulness calculation by removing the statement simplification and reason generation steps to reduce token usage and processing time

Here's a brief overview of what each metric measures:

### AnswerRelevancy
Evaluates how well the generated answer fits the given prompt. It generates virtual questions based on the context and answer, calculates vector similarity between generated and user questions, and averages the scores. Higher scores indicate better performance.

### Faithfulness
Measures the factual consistency of the generated answer compared to the given context. It segments the answer and classifies each segment as either model-generated (0) or context-based (1). The final score is the average of these classifications. Higher scores indicate better performance.

### ContextRecall
Assesses how well the retrieved context matches the LLM-generated answer. It segments the ground truth and determines if each segment can be attributed to the retrieved context. The score is the average of these attributions. Higher scores indicate better performance.

### ContextPrecision
Evaluates how relevant documents are ranked in the retrieved context. It uses an LLM to judge the usefulness of each context in the list for answer generation, with higher weights given to useful contexts appearing earlier in the list.

These modifications aim to improve evaluation accuracy, reduce processing time, and optimize resource usage while maintaining the core principles of the RAGAS evaluation framework.