
```bash
python add_book.py --scan
```


---

Explorations Report

- [x] Multi Agent Frameowrk
- [x] https://github.com/cognitivetech/ollama-ebook-summary



Agent Prompt

Prompt draft:
Roleplay as an examiner attempting to gauge if the student has read the book. based on my conversation 
Given the csv identify how much the user has read.
Update correspondingly by calling the mcp function when properly assessed the 


db model
bookid chapter

reading list
book_id, 




 Updated exploration\ollama-ebook-summary\_config.yaml with 3 additions and 3 removals        
       1    defaults:
       2      prompt: bnotes
       3 -    summary: cognitivetech/obook_summary:q3_k_m
       4 -    general: cognitivetech/obook_summary:q3_k_m
       5 -    title: cognitivetech/obook_summary:q3_k_m # cognitivetech/obook_title:q4_k_m        
         -  # default model for title generation
       3 +    summary: qwen2.5:latest       # Best balance of quality and speed for 
         + summaries
       4 +    general: qwen2.5:latest       # General-purpose tasks (cleaning, formatting)        
       5 +    title: phi3.5:latest          # Lighter model for simple title generation
       6    prompts:
       7      bnotes: # Default Prompt
       8        prompt: Write comprehensive bulleted notes summarizing the provided text,
           withd i h




● Perfect! Looking at the code (sum.py:594), the script connects to Ollama via HTTP at
  http://localhost:11434/api.

  How it Works

  1. Ollama Service
  - After installing Ollama, it automatically runs as a background service on Windows
  - You typically don't need to manually run ollama serve
  - The service starts automatically at boot
  - You can verify it's running by checking if http://localhost:11434 responds

  2. Model Loading
  - Models are NOT pre-loaded or running in memory
  - When you run python3 sum.py, it makes HTTP requests to Ollama
  - Ollama automatically loads the requested model on-demand when the script calls it
  - Only one model loads at a time as needed
  - Models unload automatically after a period of inactivity to free memory

  Quick Check

  To verify Ollama is running:

  ollama list

  If this shows your models, Ollama is already running and ready to use.

  Running the Project

  Just run the Python scripts directly:

  # Process an ebook
  python3 book2text.py yourbook.epub

  # Generate summaries
  python3 sum.py -c out/yourbook_processed.csv

  The script will automatically:
  - Connect to the Ollama service
  - Load qwen2.5:latest when it needs summaries
  - Load phi3.5:latest when it needs titles
  - Switch between models as needed

  That's it! No manual model management required.