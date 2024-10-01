import json
from datasets import Dataset
import numpy as np
import boto3
from botocore.config import Config

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
    def __init__(self, llm_id, emb_id, region):
        self.llm_id = llm_id
        self.emb_id = emb_id
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
                        "name": "StatementGenerator",
                        "description": "Generates simpler statements from paragraphs.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "paragraph_index": {
                                        "type": "integer",
                                        "description": "The index of the original paragraph"
                                    },
                                    "simpler_statements": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "description": "An array of simpler statements derived from the original paragraph"
                                    }
                                },
                                "required": ["paragraph_index", "simpler_statements"]
                            }
                        }
                    }
                },
                {
                    "toolSpec": {
                        "name": "FaithfulnessChecker",
                        "description": "Checks the faithfulness of statements based on a given context.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "statement": {
                                        "type": "string",
                                        "description": "The statement to check for faithfulness"
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "The reason for the verdict"
                                    },
                                    "verdict": {
                                        "type": "integer",
                                        "description": "1 if the statement is faithful, 0 if not"
                                    }
                                },
                                "required": ["statement", "reason", "verdict"]
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

    def generate_statements(self, question, answer):
        sys_template = """
        Given a question, an answer, and paragraphs from the answer, analyze each paragraph and break it down into one or more fully understandable statements while ensuring no pronouns are used in each statement. Use the StatementGenerator tool for each paragraph.
        """
        paragraphs = self.segment_paragraphs(answer)
        paragraphs_str = '\n'.join([f"{i}: {p}" for i, p in enumerate(paragraphs)])
        user_template = f"""
        Question: {question}
        Answer: {answer}
        Paragraphs:
        {paragraphs_str}
        Use the StatementGenerator tool for each paragraph.
        """
        sys_prompt, user_prompt = self.create_message_format(sys_template, user_template)
        response = self.converse_with_bedrock_tools(sys_prompt, user_prompt)
        output = self.parse_tool_use(response)

        statements = []
        if output:
            for item in output:
                statements.extend(item['simpler_statements'])
        return statements

    def check_faithfulness(self, context, statements):
        sys_template = """
        Your task is to judge the faithfulness of a series of statements based on given paragraphs. For each statement, use the FaithfulnessChecker tool to determine if the statement can be directly inferred from any of the paragraphs.
        """
        paragraphs = self.segment_paragraphs(context)
        paragraphs_str = json.dumps(paragraphs, ensure_ascii=False)
        statements_str = json.dumps(statements, ensure_ascii=False)
        user_template = f"""
        Paragraphs: {paragraphs_str}
        Statements: {statements_str}
        Use the FaithfulnessChecker tool for each statement.
        """
        sys_prompt, user_prompt = self.create_message_format(sys_template, user_template)
        response = self.converse_with_bedrock_tools(sys_prompt, user_prompt)
        output = self.parse_tool_use(response)

        verdicts = []
        if output:
            for item in output:
                verdicts.append(item['verdict'])
        return verdicts

    def score(self, row):
        question = row['user_input']
        answer = row['response']
        context = '\n'.join(row['retrieved_contexts'])
        statements = self.generate_statements(question, answer)
        if not statements:
            return 0.0

        verdicts = self.check_faithfulness(context, statements)
        if not verdicts:
            return 0.0

        faithful_statements = sum(verdicts)
        total_statements = len(verdicts)
        score = faithful_statements / total_statements
        return score