

# AI-Powered Interview Assistant Platform

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/langgraph-agent--framework-orange.svg)](https://github.com/langchain-ai/langgraph)

## Project Introduction

The AI-Powered Interview Assistant Platform is a comprehensive recruitment and interview support system built on large language models and multi-agent technology. This platform aims to enhance candidates' interview performance and success rates through automation and intelligence.

This system utilizes the advanced LangGraph and LlamaIndex frameworks, combined with multimodal analysis technology, to provide users with a full-cycle interview preparation service, from resume evaluation and optimization to algorithm tests and mock interviews. The project also integrates the technical requirements of the iFLYTEK A3 competition from the China Software Cup, delivering an enterprise-level interview assessment solution.

## System Architecture

### Hierarchical Architecture

This system employs an advanced hierarchical architecture, which offers the following advantages over traditional flat architectures:

- **Main Agent Connects to Subgraphs**: The main agent connects to complete subgraphs rather than single-layer nodes, enabling more complex task flow control.
- **Subgraph Collaboration**: Collaboration between the main graph and subgraphs makes the system more scalable, with more flexible and parallelizable task flows.
- **Modular Design**: Each functional module operates as an independent subgraph, facilitating maintenance and expansion.

### LangGraph Callback Mechanism

The system incorporates the LangGraph callback mechanism to track node execution status, task flow progress, and intermediate outputs:

- **Real-time Observability**: Monitor task execution in real-time for easier debugging and performance optimization.
- **State Tracking**: Accurately track the execution status and intermediate outputs of each node.
- **Enhanced Stability**: Improve the stability and monitoring of multi-node asynchronous tasks.

### Main Graph Workflow

![Main Agent Workflow](图片/main_agent_graph.png)

#### Architectural Features

The system utilizes an advanced Command-based routing and a cyclic graph architecture:

- **Command-based Routing**: Achieves precise control over node transitions using LangGraph's Command mechanism, supporting conditional logic and state updates.
- **Cyclic Graph Architecture**: Supports multi-round execution of interview processes, such as maintaining state and controlling flow in multi-stage interviews.
- **Dynamic Routing Decisions**: Dynamically determines the next agent or node to execute based on the user's selected mode and evaluation results.

## Tech Stack & Innovations

### Core Tech Stack

- **Agent Framework**: Langchain + LangGraph - For building and managing complex agent workflows.
- **State Management**: Fine-grained state definition based on TypedDict and Pydantic, providing independent input/output state modules for different agents.
- **Routing Architecture**: Command-based intelligent routing and a cyclic graph architecture for flexible task flow control.
- **Large Language Models**:
  - DeepSeek (Primary Model)
  - Qwen (Tongyi Qianwen)
  - Google Gemini (Auxiliary Model)
- **Vector Database**: Chroma DB - For vector storage and retrieval of the interview question bank.
- **Natural Language Processing**:
  - KeyBERT - For keyword extraction and matching optimization.
  - Sentence Transformers - For semantic similarity calculation.
- **Retrieval-Augmented Generation (RAG)**: LlamaIndex - To enhance interview question generation with RAG.
- **Multimodal Processing**:
  - DashScope - For speech recognition and sentiment analysis.
  - OpenCV - For video processing and body language analysis.
- **Document Processing**: python-docx, WeasyPrint - To generate professional evaluation reports and resumes.

### Core Innovations

1.  **Hierarchical Architecture**: Adopts a hierarchical structure where the main agent connects to subgraphs instead of single nodes, enhancing system scalability and task flow flexibility.
2.  **Multimodal Interview Assessment**: Combines video analysis (body language) and audio analysis (voice tone) to provide a 360-degree evaluation of interview performance.
3.  **Intelligent Question Bank System**: Built a knowledge base of over 300 real enterprise interview questions using RAG, supporting metadata filtering and re-ranking.
4.  **Parallel Processing**: Utilizes LangGraph's concurrent graph design to significantly improve the execution efficiency of tasks like resume evaluation.
5.  **Real-time Callback Monitoring**: Implements the LangGraph callback mechanism for real-time monitoring and tracking of task execution.
6.  **Fine-grained State Management**: Designs independent input/output state modules for different agents, achieving state isolation and precise control.
7.  **Intelligent Routing Architecture**: Employs Command-based routing and a cyclic graph architecture to enable flexible task flow control and support for multi-round interviews.

#### Detailed Advantages of State Management

The system uses fine-grained state definitions based on TypedDict and Pydantic, designing an independent state management mechanism for agents in different functional modules:

- **State Isolation**: Each agent has its own Input, State, and Output states, preventing state pollution and accidental modifications.
- **Type Safety**: Enforces strong type constraints through Pydantic BaseModel and TypedDict, ensuring the accuracy and consistency of state data.
- **Modular Design**: Different modules (resume evaluation, algorithm testing, interview simulation, etc.) have their own state definitions, facilitating maintenance and expansion.
- **Precise Control**: Uses `Annotated` type hints to clarify the meaning and purpose of each state field, improving code readability and maintainability.

## Core Functional Modules

### 1. Resume Evaluation Agent

- **Automated Resume Analysis**: Uses LlamaIndex's `PDFReader` to read user-uploaded resumes and splits content by headings using regular expressions.
- **Five-Dimensional Comprehensive Assessment**: Employs an asynchronous and parallel LangGraph state graph to evaluate the resume from five dimensions:
  - Future Potential Score
  - Educational Background Score
  - Tech Stack & Job Fit Score
  - Work/Internship Experience Fit Score
  - Resume Writing & Structure Score
- **Keyword Matching Optimization**: Uses the KeyBERT model for keyword extraction and matching to improve the fit between the resume and the job description.
- **Visualized Reports**: Generates a Word report containing a five-dimensional radar chart and detailed evaluation feedback.

#### Demonstration
![](图片/five_dimension_radar.png)
![](图片/b5bebb78537efbd7a995ff10bcb02dd0.png)
![](图片/b6de6ebddb938ca6abb2cfbf23739d34.png)

#### Agent Workflow
![Resume Evaluation Workflow](图片/Resume_evaluate_agent_graph.png)

### 2. Algorithm Testing Module

- **Intelligent Problem Selection**: The agent parses the target job requirements and automatically retrieves matching problems from the Codeforces platform.
- **Dynamic Difficulty Adjustment**: Dynamically adjusts problem difficulty (800-1700 rating) based on the company's reputation and the job level.
- **Automated Verification**: The system automatically checks the user's submission status on Codeforces to determine if the algorithm test is passed.
- **Concurrent Processing**: Uses LangGraph's concurrent graph design to fetch problems for multiple tags simultaneously.
#### Agent Workflow
![Algorithm Test Workflow](图片/Code_agent_graph.png)

### 3. Interview Simulation Module

- **End-to-End Simulation**: Supports a complete mock interview process from the first round (technical) to the second round (business).
- **Multi-Agent Collaboration**: Interview dialogues are conducted by multiple collaborating agents to simulate a real enterprise interview scenario.
- **Intelligent Question Generation**: Intelligently generates four types of interview questions based on the user's resume and the applied position:
  - Technical Fundamentals Questions
  - Project Experience Technical Questions
  - Business-related Questions
  - Soft Skills Questions

#### Agent Workflow
![Interview Workflow](图片/interview_agent_graph.png)

### 4. Interview Evaluation and Guidance

- **Comprehensive Evaluation System**: Assesses the quality of the user's answers from three dimensions:
  - Content Quality (technical accuracy, logic, etc.)
  - Body Language (via video analysis)
  - Voice Tone and Emotion (via audio analysis)
- **Expression Analysis**: Analyzes the user's facial expressions from the video feed and generates an expression score.
- **Personalized Guidance**: Automatically generates an interview report with standard answers, analysis of strengths and weaknesses, and suggestions for improvement.

### 5. Enhanced Interview Question Bank

- **Rich Question Bank Resources**: A collection of over 300 real enterprise interview questions covering technical, product, operations, and other roles.
- **RAG Enhancement**: Utilizes RAG technology (including metadata filtering and re-ranking) to improve the quality of generated questions.
- **Precise Matching**: Accurately matches relevant interview questions based on job information and company background, ensuring questions are targeted and practical.

### 6. Interview Summary Document Generation

- **Automated Report Generation**: The system automatically generates a detailed summary document after the interview.
- **Comprehensive Content**: Includes standard answers, the user's responses, question analysis, and evaluation feedback.
- **Professional Formatting**: Generates a professional Word report for easy review and improvement.


## Project Advantages

1.  **Full-Cycle Coverage**: Provides a complete interview preparation solution, from resume optimization to mock interviews.
2.  **Multimodal Assessment**: Offers a 360-degree evaluation of interview performance by combining video, audio, and text analysis.
3.  **Intelligent Matching**: Achieves precise matching between resumes and job positions using RAG and semantic analysis.
4.  **Highly Scalable**: The hierarchical architecture facilitates easy functional expansion and module replacement.
5.  **Real-time Monitoring**: Incorporates a callback mechanism for real-time monitoring of task execution.
6.  **Fine-grained State Isolation**: Designs independent state modules for different agents, ensuring no interference between modules and improving system stability and maintainability.
7.  **Flexible Routing Control**: Employs Command-based routing and a cyclic graph architecture to support complex multi-round interview processes and dynamic task scheduling.



## Code Structure

```
.
├── api_key.py                  # API key configuration file
├── base.py                     # Base data structures and LLM client initialization
├── requirements.txt            # Project dependency list
├── code/                       # Core code directory
│   ├── callbacks.py            # LangGraph callback implementation
│   ├── state.py                # System state definitions (TypedDict and Pydantic models)
│   ├── Main_agent.py           # Main graph agent architecture
│   ├── resume_analyse.py       # Core agent for resume analysis and evaluation
│   ├── resume_optimize.py      # Resume optimization module
│   ├── interview_agent.py      # Interview simulation agent
│   ├── generate_interview_question.py  # Interview question generation logic
│   ├── generate_question_answer.py     # Standard answer generation for interview questions
│   ├── generate_question_eval.py       # Interview answer evaluation logic
│   ├── code_test.py            # Algorithm testing agent
│   ├── rag.py                  # RAG implementation
│   ├── generate_doc.py         # Document generation (evaluation reports, interview analysis)
│   ├── do_record_video.py      # Video recording functionality
│   ├── multimoding_dispose.py  # Multimodal data processing (audio/video analysis)
│   └── stopwords-mast/         # Chinese stopwords library
├── 面试知识库/                 # Interview knowledge base (JSON format)
├── 图片/                       # System flowcharts and visual assets
├── chroma_db/                  # Chroma vector database
└── 各类输出目录/               # Various output directories (resume evaluations, videos, etc.)
```




### Core Entry Files

- `code/main.py`: The main entry point of the project, providing four functional modules: resume evaluation, resume optimization, interview training, and mock interview.
- `code/resume_agent.py`: The main agent for the resume evaluation workflow, coordinating various sub-modules.
- `code/interview_agent.py`: The interview simulation agent, managing the entire interview process.
- `code/state.py`: The system state definition module, providing fine-grained state management and isolation for different agents.

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- A GPU with CUDA support (recommended for accelerating model inference)

### Installation Steps

1.  **Clone the project**:
    ```bash
    git clone <project_repository_url>
    cd <project_directory>
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # or venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API keys**:
    Fill in your API keys in the `api_key.py` file:
    ```python
    dp_api = "YOUR_DEEPSEEK_API_KEY"
    search_api_key = "YOUR_SERPER_API_KEY"  # www.serper.dev
    google_api = "YOUR_GOOGLE_API_KEY"
    qwen_api = "YOUR_QWEN_API_KEY"
    ```

5.  **Download model files**:
    - Download the sentence-transformer model (e.g., `shibing624/text2vec-base-chinese`)
    - Download the embedding model
    - Download the rerank model (e.g., `BAAI/bge-reranker-base`)

6.  **Create necessary directories**:
    The system will create the required directories automatically, but you can also check them manually:
    `pdf_reports`, `resume_data`, `video_picture`, `简历评估` (resume_evaluation), `简历照片` (resume_photos), `雷达图` (radar_charts), `面试视频（用户）` (user_interview_videos), `面试知识库` (interview_knowledge_base), `问题解析` (question_analysis), `优化简历` (optimized_resumes), `语音资料（用户）` (user_audio_files).

7.  **Run an example**:
    Modify the `path` and `job` variables in `code/main.py` and uncomment the function you want to run:
    ```bash
    python code/main.py
    ```

## Usage Instructions

- **Resume Evaluation**: Run the `main1()` function in `main.py`.
- **Resume Optimization**: Run the `main2()` function in `main.py`.
- **Interview Training**: Run the `main3()` function in `main.py`. You can adjust the `interview_question_num` parameter.
- **Mock Interview (with Algorithm Test)**: Run the `main4()` function in `main.py`. You need to provide your Codeforces account handle.