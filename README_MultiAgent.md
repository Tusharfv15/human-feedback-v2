# Multi-Agent Semiconductor Processing Workflow

This project implements a multi-agent workflow for semiconductor processing using LangGraph and a finetuned LLM. The system intelligently routes user queries to specialized agents and handles dependencies between different processing steps.

## Architecture

### Agents

1. **Doping Agent** (`doping-agent`)
   - Handles ion implantation, dopant activation, junction formation
   - Programs: `programs/doping-programs.json`
   - Knowledge: `knowledge/doping-knowledge.txt`

2. **Etching Agent** (`etching-agent`)
   - Handles plasma etching, pattern transfer, material removal
   - Programs: `programs/etching-programs.json`
   - Knowledge: `knowledge/etching-knowledge.txt`

3. **Wafer Agent** (`wafer-agent`)
   - Handles wafer cleaning, oxidation, CVD, packaging
   - Programs: `programs/wafer-programs.json`
   - Knowledge: `knowledge/wafer-knowledge.txt`

### Orchestrator

The orchestrator is a finetuned LLM that:
- Routes incoming queries to appropriate agents
- Analyzes dependencies between processing steps
- Manages the flow between agents
- Handles human feedback integration

## Workflow

1. **Query Routing**: User query → Orchestrator decides which agent to use
2. **Agent Processing**: Selected agent finds relevant program or uses knowledge base
3. **Solution Generation**: Agent generates processing parameters and recommendations
4. **Dependency Analysis**: Orchestrator checks if other agents are needed
5. **Agent Chaining**: If dependencies exist, route to next required agent
6. **Human Feedback**: Present solution to user for approval or modifications
7. **Iteration**: If feedback provided, refine solution and re-analyze dependencies
8. **Finalization**: Present final multi-agent solution

## Files Structure

```
├── multi_agent_workflow.py       # Main multi-agent workflow implementation
├── doping_workflow.py            # Original single-agent workflow (legacy)
├── example_multi_agent_usage.py  # Example usage scenarios
├── model_manager.py              # LLM model management
├── programs/
│   ├── doping-programs.json      # Doping process programs
│   ├── etching-programs.json     # Etching process programs
│   └── wafer-programs.json       # Wafer processing programs
├── knowledge/
│   ├── doping-knowledge.txt      # Doping expertise knowledge base
│   ├── etching-knowledge.txt     # Etching expertise knowledge base
│   └── wafer-knowledge.txt       # Wafer processing knowledge base
└── README_MultiAgent.md          # This file
```

## Usage

### Basic Usage

```python
from multi_agent_workflow import run_multi_agent_workflow

# Run a query through the multi-agent system
query = "I need to create shallow p+ junctions with minimal thermal budget"
result = run_multi_agent_workflow(query)
```

### Example Queries

**Doping-focused query:**
```python
query = "I need boron implantation for p-type source/drain with rapid thermal annealing"
```

**Etching-focused query:**
```python
query = "I need to etch polysilicon gates with vertical sidewalls and high selectivity"
```

**Wafer processing query:**
```python
query = "I need RCA cleaning sequence followed by thermal oxidation for 10nm gate oxide"
```

**Multi-agent query:**
```python
query = "I need complete CMOS fabrication with shallow junctions, selective etching, and proper cleaning"
```

## Key Features

### Intelligent Routing
- LLM-based query analysis determines optimal agent
- Reasoning provided for routing decisions
- Fallback mechanisms for edge cases

### Dependency Management
- Automatic analysis of solution dependencies
- Cross-agent parameter requirements detected
- Agent chaining for complex processes

### Human Feedback Loop
- Interactive solution review and modification
- Iterative refinement based on user input
- Comprehensive change tracking

### Flexible Program Selection
- JSON-based program databases for each agent
- LLM selects most relevant programs
- Knowledge base fallback when no program matches

### Robust Error Handling
- GPU memory management
- Graceful fallbacks for LLM failures
- Comprehensive logging and debugging

## Agent Capabilities

### Doping Agent
- Ion implantation parameter optimization
- Dopant activation and annealing schedules
- Junction depth and concentration control
- Thermal budget management

### Etching Agent  
- Plasma chemistry optimization
- Selectivity and anisotropy control
- Critical dimension management
- Multi-material etch sequences

### Wafer Agent
- Surface cleaning and preparation
- Thermal oxidation processes
- CVD thin film deposition
- Wafer-level packaging considerations

## Human Feedback Options

During the workflow, users can:
- **approve**: Accept the current solution
- **modify:<feedback>**: Provide specific modification requests
- **exit**: Finalize the current solution

Example feedback:
```
modify:reduce thermal budget by 20% and increase selectivity
modify:change from wet to dry etching for better anisotropy
modify:add additional cleaning step before oxidation
```

## Running Examples

```bash
python example_multi_agent_usage.py
```

This will present options to test different agent routing scenarios and multi-agent workflows.

## Dependencies

- LangGraph: Workflow orchestration
- LangSmith: Tracing and monitoring  
- PyTorch: LLM inference
- Transformers: Model loading and tokenization

## Configuration

The system uses environment variables for configuration:
- Model paths and parameters in `model_manager.py`
- LangSmith tracing configuration
- GPU/CPU inference settings

## Monitoring

The workflow includes comprehensive tracing through LangSmith:
- Agent routing decisions
- Program selection reasoning
- Dependency analysis results
- Solution generation metrics
- Human feedback interactions

This enables detailed analysis of workflow performance and optimization opportunities.