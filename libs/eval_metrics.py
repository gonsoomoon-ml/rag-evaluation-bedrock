import json
from datasets import Dataset
import numpy as np
import boto3
from botocore.config import Config
import re

class AnswerRelevancy:
    def __init__(self, llm_id, emb_id, region, strictness=3):
        self.llm_id = llm_id
        self.emb_id = emb_id
        self.region = region
        self.strictness = strictness
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.boto3_client = boto3.client("bedrock-runtime", config=retry_config)
        self.tool_config = self.init_tool()

    def init_tool(self):
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "QuestionGenerator",
                        "description": "Generates questions based on the given context and answer.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The generated question"
                                    },
                                    "noncommittal": {
                                        "type": "string",
                                        "description": "Give 'noncommittal' as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, 'I don't know' or 'I'm not sure' are noncommittal answers."
                                    }
                                },
                                "required": ["question", "answer"]
                            }
                        }
                    }
                }
            ]
        }
        return tool_config

    def create_message_format(self, sys_template, user_template):
        sys_prompt = [{"text": sys_template}]
        usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
        return sys_prompt, usr_prompt

    def converse_with_bedrock_tools(self, sys_prompt, usr_prompt):
        inference_config = {"temperature": 0.0, "topP": 0.1}
        response = self.boto3_client.converse(
            modelId=self.llm_id,
            messages=usr_prompt,
            system=sys_prompt,
            toolConfig=self.tool_config,
            inferenceConfig=inference_config
        )
        return response

    def parse_tool_use(self, message):
        stop_reason = message['stopReason']

        if stop_reason == 'tool_use':
            tool_requests = message['output']['message']['content']
            for tool_request in tool_requests:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']

                    if tool['name'] == 'QuestionGenerator':
                        return tool['input']
        return None

    def generate_questions(self, answer, context):
        sys_template = """
        Generate a question for the given answer based on the given context and identify if the answer is noncommittal. 
        """

        user_template = f"""
        Answer: {answer}
        Context: {context}
        Use 'QuestionGenerator' tool to generate a question.
        """

        questions = []
        noncommittals = []

        for _ in range(self.strictness):
            sys_prompt, user_prompt = self.create_message_format(sys_template, user_template)
            response = self.converse_with_bedrock_tools(sys_prompt, user_prompt)
            output = self.parse_tool_use(response)          
            
            question = output['question']
            noncommittal = int(output['noncommittal'])

            questions.append(question)
            noncommittals.append(noncommittal)
        
        return questions, noncommittals        

    def get_embedding_vector(self, text):
        request = json.dumps({"inputText": text})
        response = self.boto3_client.invoke_model(modelId=self.emb_id, body=request)
        embedding = json.loads(response["body"].read())["embedding"]
        return embedding

    def score(self, row):
        user_input = row['user_input']
        answer = row['response']
        context = row['retrieved_contexts']
        context_str = '\n'.join(context)

        generated_questions, noncommittals = self.generate_questions(answer, context_str)

        user_input_vec = self.get_embedding_vector(user_input)
        generated_vectors = [self.get_embedding_vector(q) for q in generated_questions]
        similarities = [
            self.cosine_similarity(user_input_vec, vec) for vec in generated_vectors
        ]
        avg_similarity = np.mean(similarities)
        is_committal = all(not nc for nc in noncommittals)

        return avg_similarity * (1 if is_committal else 0)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class Faithfulness:
    def __init__(self, llm_id, region):
        self.llm_id = llm_id
        self.region = region
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.boto3_client = boto3.client("bedrock-runtime", config=retry_config)
        self.tool_config = self.init_tool()

    def init_tool(self):
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "FaithfulnessChecker",
                        "description": "Checks the faithfulness of paragraphs based on a given context.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "verdicts": {
                                        "type": "array",
                                        "items": {
                                            "type": "integer",
                                            "enum": [0, 1]
                                        },
                                        "description": "Array of 0 (not faithful) or 1 (faithful) for each paragraph"
                                    }
                                },
                                "required": ["verdicts"]
                            }
                        }
                    }
                }
            ]
        }
        return tool_config

    def create_message_format(self, sys_template, user_template):
        sys_prompt = [{"text": sys_template}]
        usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
        return sys_prompt, usr_prompt

    def converse_with_bedrock_tools(self, sys_prompt, usr_prompt):
        inference_config = {"temperature": 0.0, "topP": 0.1}
        response = self.boto3_client.converse(
            modelId=self.llm_id,
            messages=usr_prompt,
            system=sys_prompt,
            toolConfig=self.tool_config,
            inferenceConfig=inference_config
        )
        return response

    def parse_tool_use(self, message):
        stop_reason = message['stopReason']
        if stop_reason == 'tool_use':
            tool_requests = message['output']['message']['content']
            results = []
            for tool_request in tool_requests:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']
                    results.append(tool['input'])
            return results
        return None

    def segment_paragraphs(self, text):
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def check_faithfulness(self, context, user_input):
        sys_template = """
        Your task is to judge the faithfulness of a series of paragraphs based on a given context. For each paragraph, determine if it can be directly inferred from the context..
        """
        paragraphs = self.segment_paragraphs(user_input)
        paragraphs_str = '\n\n'.join([f"Paragraph {i}:\n {p}" for i, p in enumerate(paragraphs)])
        user_template = f"""
        Context: {context}

        Paragraphs:
        {paragraphs_str}

        Use the FaithfulnessChecker tool to evaluate the given paragraphs.
        """
        sys_prompt, user_prompt = self.create_message_format(sys_template, user_template)
        response = self.converse_with_bedrock_tools(sys_prompt, user_prompt)
        output = self.parse_tool_use(response)

        if output and len(output) > 0:
            return output[0]['verdicts']
        return []

    def score(self, row):
        context = row['retrieved_contexts']
        user_input = row['response']
        verdicts = self.check_faithfulness(context, user_input)
        print(verdicts)
        if not verdicts:
            return 0.0

        faithful_paragraphs = sum(verdicts)
        total_paragraphs = len(verdicts)
        score = faithful_paragraphs / total_paragraphs

        return score


class ContextRecall:
    def __init__(self, llm_id, region):
        self.llm_id = llm_id
        self.region = region
        retry_config = Config(
            region_name=region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.boto3_client = boto3.client("bedrock-runtime", config=retry_config)
        self.tool_config = self.init_tool()

    def init_tool(self):
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "ContextRecallClassifier",
                        "description": "Classifies if a statement can be attributed to the given context.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "attributed": {
                                        "type": "array",
                                        "items": {
                                            "type": "integer",
                                            "enum": [0, 1]
                                        },
                                        "description": "Array of 0 (not attributed) or 1 (attributed) for each statement"
                                    }
                                },
                                "required": ["attributed"]
                            }
                        }
                    }
                }
            ]
        }
        return tool_config

    def create_message_format(self, sys_template, user_template):
        sys_prompt = [{"text": sys_template}]
        usr_prompt = [{"role": "user", "content": [{"text": user_template}]}]
        return sys_prompt, usr_prompt

    def converse_with_bedrock_tools(self, sys_prompt, usr_prompt):
        inference_config = {"temperature": 0.0, "topP": 0.1}
        response = self.boto3_client.converse(
            modelId=self.llm_id,
            messages=usr_prompt,
            system=sys_prompt,
            toolConfig=self.tool_config,
            inferenceConfig=inference_config
        )
        return response

    def parse_tool_use(self, message):
        stop_reason = message['stopReason']
        if stop_reason == 'tool_use':
            tool_requests = message['output']['message']['content']
            results = []
            for tool_request in tool_requests:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']
                    results.append(tool['input'])
            return results
        return None

    def segment_paragraphs(self, text):
        paragraphs = re.split(r'\n{2,}|\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def check_context_recall(self, contexts, reference):
        sys_template = """
        Given multiple contexts and a reference answer, analyze each statement in the reference and classify if it can be attributed to any of the given contexts.
        """
        paragraphs = self.segment_paragraphs(reference)
        paragraphs_str = '\n\n'.join([f"Paragraph {i+1}: {p}" for i, p in enumerate(paragraphs)])
        contexts_str = '\n'.join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
        user_template = f"""
        Contexts:
        {contexts_str}

        Reference paragraphs:
        {paragraphs_str}

        Use the ContextRecallClassifier tool to evaluate each paragraph in the reference.
        """
        sys_prompt, user_prompt = self.create_message_format(sys_template, user_template)
        response = self.converse_with_bedrock_tools(sys_prompt, user_prompt)
        output = self.parse_tool_use(response)

        if output and len(output) > 0:
            return output[0]['attributed']
        return []

    def score(self, row):
        contexts = row['retrieved_contexts']
        reference = row['reference']
        attributed = self.check_context_recall(contexts, reference)
        print(attributed)
        if not attributed:
            return 0.0

        total_paragraphs = len(attributed)
        attributed_paragraphs = sum(attributed)
        score = attributed_paragraphs / total_paragraphs if total_paragraphs > 0 else 0.0

        return score