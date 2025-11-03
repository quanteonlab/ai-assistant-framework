# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 3)

**Starting Chapter:** Foundation Model Use Cases

---

#### AI Use Case Categories Overview
Background context: The text discusses various categorizations of AI use cases, highlighting how different organizations and surveys define these categories. This is important for understanding where to apply foundation models effectively.

:p What are some ways different organizations categorize AI use cases?
??x
Organizations like AWS, O'Reilly, Deloitte, and Gartner provide diverse categorizations based on their industry focus and value capture strategies. For example:
- **AWS**: Customer experience, employee productivity, process optimization.
- **O'Reilly (2024)**: Programming, data analysis, customer support, marketing copy, other copy, research, web design, art.
- **Deloitte**: Cost reduction, process efficiency, growth, accelerating innovation.
- **Gartner**: Business continuity.

These categorizations help in aligning AI applications with specific business needs and objectives. For instance, Gartner categorizes use cases based on the potential impact of not adopting generative AI, where 7% cite business continuity as a key driver.

x??

---

#### Exposure to AI by Occupations
Background context: The text mentions an Eloundou et al. (2023) study that defines tasks and occupations exposed to AI if they can reduce task completion time by at least 50%. This helps in identifying which jobs might be most affected or can benefit from AI integration.

:p According to the Eloundou et al. (2023) study, what are some occupations with high exposure to AI?
??x
According to the Eloundou et al. (2023) study:
- **High Exposure**: Interpreters and translators, survey researchers, poets, lyricists, and creative writers, animal scientists, public relations specialists.
  - These have at least 76.5% exposure.

- **Human β**: Survey researchers, writers and authors, interpreters and translators, public relations specialists, animal scientists.
  - This category includes a bit lower but still significant exposure (80.6% to 84.4%).

- **Human ζ**: Mathematicians, tax preparers, financial quantitative analysts, writers and authors, web and digital interface designers.
  - These are fully exposed with 100% exposure.

x??

---

#### Common Generative AI Use Cases
Background context: The text categorizes common generative AI use cases into eight groups across consumer and enterprise applications. This is crucial for understanding the diverse areas where foundation models can be applied effectively.

:p What categories do common generative AI use cases fall under according to the text?
??x
Common generative AI use cases are categorized into:
1. **Coding**
2. **Image and video production** (Photo and video editing, design, presentation)
3. **Ad generation**
4. **Writing** (Email, social media and blog posts, copywriting, SEO reports, memos, design docs)
5. **Education** (Tutoring, essay grading, employee onboarding, upskilling training)
6. **Conversational bots** (General chatbot, AI companion, customer support, product copilots)
7. **Information aggregation and summarization**
8. **Market research** (Data organization, image search, memex knowledge management, document processing, workflow automation, travel planning, event planning, lead generation)

These categories provide a broad framework for applying foundation models to real-world problems.

x??

---

#### Distribution of Use Cases in Open Source Applications
Background context: The text discusses the distribution of AI use cases across 205 open source applications. This information is valuable for understanding which areas are more active and where there might be opportunities for innovation.

:p According to the analysis of 205 open source repositories, how is the distribution of generative AI use cases?
??x
The distribution among the 205 open source repositories on GitHub shows a varied spread:
- **Coding**: Most common.
- **Image and video production**: Well-represented.
- **Ad generation** and **writing**: Moderate presence.
- **Education**, **conversational bots**, **information aggregation and summarization**, **market research**: Less prevalent, but still represented.

This distribution suggests that certain areas might be more saturated than others. Builders of applications in less commonly covered domains (like education) might find these areas more suitable for enterprise use cases due to lower competition.

x??

---

#### Enterprise vs Consumer Applications
Background context: The text highlights the differences between enterprise and consumer AI applications, noting that enterprises generally prefer lower-risk internal-facing applications over external-facing ones. This understanding is crucial for aligning AI initiatives with organizational goals.

:p What are some key differences noted in the text between enterprise and consumer AI applications?
??x
Key differences include:
- **Risk Tolerance**: Enterprises favor low-risk internal applications (e.g., knowledge management) over high-risk external applications (e.g., customer support chatbots).
- **Deployment Speed**: Internal applications are deployed faster due to lower risks.
- **Application Complexity**: Many enterprise applications remain close-ended, like classification tasks, which are easier to evaluate and risk-manage.

These differences influence strategic decisions on where to deploy AI resources within an organization.

x??

---

#### AI Coding Tools Popularity and Success
Background context: The text discusses the increasing popularity of AI coding tools, with specific examples like GitHub Copilot, Magic, and Anysphere. It highlights the rapid growth and significant funding these tools have received. 
:p How has the success of AI coding tools been demonstrated?
??x
The success of AI coding tools is evidenced by their widespread adoption and substantial financial backing. For instance, GitHub Copilot achieved an annual recurring revenue of $100 million within two years after its launch. Additionally, Magic and Anysphere raised large amounts of funding—$320 million for Magic and $60 million for Anysphere in August 2024.
x??

---
#### Code Completion Tools
Background context: The text mentions the success of GitHub Copilot as a code completion tool, which is one of the earliest successes of foundation models in production. It illustrates how these tools can significantly enhance developer productivity.
:p What are some popular AI coding tools that focus on code completion?
??x
Popular AI coding tools for code completion include GitHub Copilot, Magic, and Anysphere. These tools help developers write code faster by suggesting completions based on existing codebases or patterns.
x??

---
#### General Coding Tools
Background context: The text lists several general-purpose AI coding tools that aid in various tasks such as data extraction, English-to-code conversion, design-to-code generation, language translation, documentation writing, test creation, and commit message generation.
:p Name some of the general AI-powered coding tools mentioned in the text.
??x
The general AI-powered coding tools mentioned include AgentGPT for structured data extraction, DB-GPT and SQL Chat for English to code conversion, screenshot-to-code and draw-a-ui for generating frontend code from designs, GPT-Migrate and AI Code Translator for language translation, Autodoc for documentation writing, PentestGPT for creating tests, and AI Commits for generating commit messages.
x??

---
#### Frontend vs. Backend Development
Background context: The text notes that developers have observed AI being better at frontend development than backend development based on their experiences with AI coding tools.
:p According to the text, how do developers perceive AI's performance in frontend versus backend development?
??x
Developers have noticed that AI is much better at frontend development compared to backend development. This perception comes from observing the effectiveness of AI coding tools in tasks like generating frontend code from designs or screenshots but less so in complex backend tasks.
x??

---
#### Developer Productivity with AI Coding Tools
Background context: The text highlights that AI can significantly increase developer productivity for simpler tasks, such as documentation and code generation. However, it notes that the impact on highly complex tasks is minimal.
:p How does AI affect developer productivity according to the text?
??x
AI can help developers be twice as productive for documentation and 25–50 percent more productive for code generation and code refactoring. However, minimal improvement in productivity was observed for highly complex tasks. This indicates that while AI can greatly enhance efficiency in simple coding activities, it has a limited impact on extremely intricate development challenges.
x??

---
#### Future of Software Engineering
Background context: The text presents contrasting views about the future role of AI in software engineering—from complete replacement to mere augmentation—highlighting the potential for significant changes in developer roles and responsibilities.
:p What are some predictions regarding AI's role in software engineering?
??x
There are two contrasting views on AI's role in software engineering. On one end, NVIDIA CEO Jensen Huang predicts that AI will replace human software engineers. On the other end, many developers believe they will never be replaced by AI due to technical and emotional reasons.
x??

---
#### Productivity Improvement Across Tasks
Background context: The text provides a McKinsey study indicating different levels of productivity improvement for various tasks when using AI coding tools.
:p What does the McKinsey study indicate about the impact of AI on developer productivity?
??x
The McKinsey study shows that AI can help developers be significantly more productive, especially for simple tasks such as documentation and code generation. The productivity gains were around 25–50 percent for code generation and refactoring but minimal for highly complex tasks.
x??

---

#### Marketing and Advertising Automation
Background context explaining how AI can reduce costs by automating marketing activities. Mention that on average, 11 percent of a company's budget is spent on marketing.
:p How does AI help with marketing and advertising?
??x
AI can automate various aspects of marketing and advertising, leading to significant cost savings. For instance, companies can use AI to generate promotional images and videos automatically, brainstorm ideas for ads, or create multiple ad drafts to test different variations. This automation allows businesses to achieve more with a smaller budget.
???x

#### Creative Applications of AI
Background context on the success of AI in creative tasks like image generation, video production, etc., citing examples such as Midjourney and Adobe Firefly.
:p What are some applications of AI in creative industries?
??x
AI is particularly effective for creative tasks due to its probabilistic nature. Notable examples include:
- **Midjourney**: An AI startup that generates $200 million annually in recurring revenue through image generation services.
- **Adobe Firefly**: Provides photo editing features powered by AI.
- **Runway, Pika Labs, and Sora**: Offer video generation capabilities with AI.
???x

---

#### AI in Writing and Content Generation
Background context on how AI aids writing, mentioning autocorrect, auto-completion, and the MIT study that evaluated ChatGPT's impact. Include an example of AI’s use in generating text.
:p How does AI assist in writing?
??x
AI can significantly aid the writing process by suggesting phrases, completing sentences, and even generating entire paragraphs or sections of content. For instance, AI models like ChatGPT have been found to reduce the time taken for tasks by 40% while improving output quality by 18%. This means that AI is particularly beneficial for writers who may struggle with writing.
???x

---

#### Enterprise Use Cases
Background context on how enterprises are using AI in marketing, sales, and team communication. Mention tools like HubSpot and Salesforce.
:p What are some enterprise use cases of AI?
??x
Enterprises can leverage AI across various departments:
- **Marketing**: Automating ad generation, brainstorming ideas, and creating variations based on seasons or locations.
- **Sales and Communication**: Writing performance reports, crafting cold outreach emails, and generating product descriptions.
Tools like HubSpot and Salesforce provide enterprise users with built-in AI capabilities to enhance web content and communication strategies.
???x

---

#### AI in Consumer Applications
Background context on how consumers use AI for better communication, writing essays, etc. Include examples of startups using AI to generate books.
:p How do consumers use AI?
??x
Consumers benefit from AI through various applications:
- **Improved Communication**: Tools can help draft emails or messages.
- **Essay Writing and Book Generation**: AI assists students in writing essays and even generates entire books across genres like children’s, fan fiction, romance, and fantasy. These books can be interactive based on reader preferences.
???x

---

#### Grammar Checking
Background context on how AI helps with grammar checking and improving coherence. Mention Grammarly as an example.
:p How does AI help in writing?
??x
AI tools like Grammarly use advanced models to refine users' writing, making it more fluent, coherent, and clear by suggesting edits or improvements.
???x

