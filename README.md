# Easy MIPROv2
Companies often deploy their own internal version of ChatGPT for their employees. There are a variety of reasons why: security, the ability to connect to internal databases, etc. One improvement that can be easily done is to tune the system prompt of the internal chatbot to better suit the unique needs of the company's employees.

This repo illustrates how to refine your system prompt given typical collected user feedback. It uses DSPy + MIPROv2 to examine upvoted question/answer pairs from historical chat sessions and optimizes the system prompt accordingly.

## Features
- Uses MIPROv2 for prompt optimization
- Combines ROUGE-L and LLM-as-judge for evaluation
- Works with historical chat data and user feedback
- Generates optimized system prompts for your specific use case

## Prerequisites
- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation
1. Clone this repository:
```bash
git clone https://github.com/jeffleft/easy-prompt-optimization.git
cd easy-prompt-optimization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## How to Run
1. Prepare your data:
   - Place your conversation data in `data/conversations.jsonl`
   - Place your feedback data in `data/feedback.json`

2. Run the training script:
```bash
python generic_train.py
```

The script will:
- Load and process your conversation data
- Split the data into training and validation sets
- Use MIPROv2 to optimize the system prompt
- Save the optimized prompt to `basic_prompt.json`
- Print the final system prompt

## Data Format
### Conversations
The conversations should be in JSONL format with the following structure:
```json
{
  "id": "unique_conversation_id",
  "messages": [
    {"role": "user", "content": "user message"},
    {"role": "assistant", "content": "assistant response"}
  ]
}
```

### Feedback
The feedback file should be a JSON file with the following structure:
```json
{
  "conversation_id_message_index": {
    "feedback": "positive"  // or "negative"
  }
}
```

## Synthetic Data
To illustrate how this library works, I generated sample conversations between the employees at the fictional QuantGlyph company and an internal chatbot.

**Data Generation Process**:
   - Used GPT to simulate conversations between employees and the chatbot
   - Created diverse employee personas with different roles and expertise levels
   - Generated conversations covering various company products and workflows
   - Included both technical and non-technical queries
   - Simulated multi-turn conversations with natural progression
   - Used a streamlit app to rate the chatbot responses manually

The synthetic data serves as a realistic example of how the system can be used to optimize prompts for a specific company's needs. You can find the generated conversations in the `data/` directory and use them as a template for your own data collection.

## Enterprise Assistant Dashboard
When deploying an internal ChatGPT, companies can benefit from a high-level view of the performance of the assistant, as well as an understanding of potential productivity gains. We include a simple demo of how a dashboard for monitoring and analyzing the usage of the assistant based on chat logs might function.

![Dashboard Screenshot](data/images/Screenshot%202025-05-09%20at%202.48.20%20PM.png)

Here you can see productivity gains estimated from the type of task being accomplished in each (successful) conversation:

![Detailed Metrics](data/images/Screenshot%202025-05-09%20at%202.48.28%20PM.png)

To access the dashboard, run:
```bash
streamlit run st_o11y.py
```

## License
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
